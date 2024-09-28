import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib.pyplot as plt
import numpy as np
from .guided_diffusion.guided_diffusion.imagen import PerceiverResampler

import matplotlib.pyplot as plt

__version__ = "0.0"

import os

from pynvml import *

from vector_quantize_pytorch import ResidualVQ
import yaml
from .vqvae.vqvae import EncoderMLP
import einops
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Model
from transformers import GPT2Config
from .img_encoder import Encoder
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

import tensorboard as tb

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")

def tensors2vectors(tensors):
    def tensor2vector(tensor):
        flo = (tensor.permute(1, 2, 0).numpy()-0.5)*1000
        r = 8
        plt.quiver(flo[::-r, ::r, 0], -flo[::-r, ::r, 1], color='r', scale=r*20)
        plt.savefig('temp.jpg')
        plt.clf()
        return plt.imread('temp.jpg').transpose(2, 0, 1)
    return torch.from_numpy(np.array([tensor2vector(tensor) for tensor in tensors])) / 255

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor): 
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


   
class GoalGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps = 1000,
        sampling_timesteps = 100,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        # assert not (type(self) == GoalGaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise): # x_t = sqrt{alpha_cumprod_t}*x_0 + sqrt{1 - alpha_cumprod_t}*epsilon, return x_0
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0): # x_t = sqrt{alpha_cumprod_t}*x_0 + sqrt{1 - alpha_cumprod_t}*epsilon, return epsilon
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        ) # x_{t-1}的均值
        posterior_variance = extract(self.posterior_variance, t, x_t.shape) # x_{t-1}的方差
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_cond, task_embed,  clip_x_start=False, rederive_pred_noise=False, guidance_weight=0):
        # task_embed = self.text_encoder(goal).last_hidden_state
        model_output = self.model(torch.cat([x, x_cond], dim=1), t, task_embed) # x.shape: (b, (f c), h, w), x_cond.shape: (b, c, h, w)
        if guidance_weight > 0.0:
            uncond_model_output = self.model(torch.cat([x, x_cond], dim=1), t, task_embed*0.0)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity # partial: 给定部分参数，创建一个新的函数

        if self.objective == 'pred_noise':# 预测噪声法：噪声直接预测，x_0从噪声计算得到
            if guidance_weight == 0:
                pred_noise = model_output
            else:
                pred_noise = (1 + guidance_weight)*model_output - guidance_weight*uncond_model_output # classifier-free guidance diffusion

            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0': # 模型预测 x_start, x_start -> noise (-> x_start, if guidance_weight!=0)
            x_start = model_output
            x_start = maybe_clip(x_start)

            if guidance_weight == 0:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                uncond_x_start = uncond_model_output
                uncond_x_start = maybe_clip(uncond_x_start)
                cond_noise = self.predict_noise_from_start(x, t, x_start)
                uncond_noise = self.predict_noise_from_start(x, t, uncond_x_start)
                pred_noise = (1 + guidance_weight)*cond_noise - guidance_weight*uncond_noise # classifier-free guidance diffusion
                x_start = self.predict_start_from_noise(x, t, pred_noise)
            
        elif self.objective == 'pred_v': # 模型预测 v, v -> x_start -> noise (-> x_start, if guidance_weight!=0)
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            
            if guidance_weight == 0:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                uncond_v = uncond_model_output
                uncond_x_start = self.predict_start_from_v(x, t, uncond_v)
                uncond_noise = self.predict_noise_from_start(x, t, uncond_x_start)
                cond_noise = self.predict_noise_from_start(x, t, x_start)
                pred_noise = (1 + guidance_weight)*cond_noise - guidance_weight*uncond_noise
                x_start = self.predict_start_from_noise(x, t, pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond, task_embed,  clip_denoised=False, guidance_weight=0):  # 模型预测x_0, 根据模型预测的x_0计算后验分布
        preds = self.model_predictions(x, t, x_cond, task_embed, guidance_weight=guidance_weight)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_cond, task_embed, guidance_weight=0):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, x_cond, task_embed, clip_denoised = True, guidance_weight=guidance_weight)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, task_embed, return_all_timesteps=False, guidance_weight=0): # DDPM采样过程
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, x_cond, task_embed, guidance_weight=guidance_weight)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, x_cond, task_embed, return_all_timesteps=False, guidance_weight=0): # DDIM采样过程
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            # self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, task_embed, clip_x_start = False, rederive_pred_noise = True, guidance_weight=guidance_weight)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, x_cond, task_embed, batch_size = 16, return_all_timesteps = False, guidance_weight=0):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size[0], image_size[1]), x_cond, task_embed,  return_all_timesteps = return_all_timesteps, guidance_weight=guidance_weight)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5): # 接受x1, x2, 加噪后再去噪
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, x_cond, task_embed, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step

        model_out = self.model(torch.cat([x, x_cond], dim=1), t, task_embed)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, img_cond, task_embed):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}, got({h}, {w})'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, img_cond, task_embed)


class NewGoalGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps = 1000,
        sampling_timesteps = 100,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        # assert not (type(self) == GoalGaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise): # x_t = sqrt{alpha_cumprod_t}*x_0 + sqrt{1 - alpha_cumprod_t}*epsilon, return x_0
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0): # x_t = sqrt{alpha_cumprod_t}*x_0 + sqrt{1 - alpha_cumprod_t}*epsilon, return epsilon
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        ) # x_{t-1}的均值
        posterior_variance = extract(self.posterior_variance, t, x_t.shape) # x_{t-1}的方差
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_cond, task_embed,  clip_x_start=False, rederive_pred_noise=False, guidance_weight=0):
        # task_embed = self.text_encoder(goal).last_hidden_state
        model_output = self.model(torch.cat([x, x_cond], dim=1), t, task_embed) # x.shape: (b, (f c), h, w), x_cond.shape: (b, c, h, w)
        if guidance_weight > 0.0:
            uncond_model_output = self.model(torch.cat([x, x_cond], dim=1), t, task_embed*0.0)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity # partial: 给定部分参数，创建一个新的函数

        if self.objective == 'pred_noise':# 预测噪声法：噪声直接预测，x_0从噪声计算得到
            if guidance_weight == 0:
                pred_noise = model_output
            else:
                pred_noise = (1 + guidance_weight)*model_output - guidance_weight*uncond_model_output # classifier-free guidance diffusion

            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0': # 模型预测 x_start, x_start -> noise (-> x_start, if guidance_weight!=0)
            x_start = model_output
            x_start = maybe_clip(x_start)

            if guidance_weight == 0:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                uncond_x_start = uncond_model_output
                uncond_x_start = maybe_clip(uncond_x_start)
                cond_noise = self.predict_noise_from_start(x, t, x_start)
                uncond_noise = self.predict_noise_from_start(x, t, uncond_x_start)
                pred_noise = (1 + guidance_weight)*cond_noise - guidance_weight*uncond_noise # classifier-free guidance diffusion
                x_start = self.predict_start_from_noise(x, t, pred_noise)
            
        elif self.objective == 'pred_v': # 模型预测 v, v -> x_start -> noise (-> x_start, if guidance_weight!=0)
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            
            if guidance_weight == 0:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                uncond_v = uncond_model_output
                uncond_x_start = self.predict_start_from_v(x, t, uncond_v)
                uncond_noise = self.predict_noise_from_start(x, t, uncond_x_start)
                cond_noise = self.predict_noise_from_start(x, t, x_start)
                pred_noise = (1 + guidance_weight)*cond_noise - guidance_weight*uncond_noise
                x_start = self.predict_start_from_noise(x, t, pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond, task_embed,  clip_denoised=False, guidance_weight=0):  # 模型预测x_0, 根据模型预测的x_0计算后验分布
        preds = self.model_predictions(x, t, x_cond, task_embed, guidance_weight=guidance_weight)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_cond, task_embed, guidance_weight=0):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, x_cond, task_embed, clip_denoised = True, guidance_weight=guidance_weight)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, task_embed, return_all_timesteps=False, guidance_weight=0): # DDPM采样过程
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, x_cond, task_embed, guidance_weight=guidance_weight)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, x_cond, task_embed, return_all_timesteps=False, guidance_weight=0): # DDIM采样过程
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            # self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, task_embed, clip_x_start = False, rederive_pred_noise = True, guidance_weight=guidance_weight)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, x_cond, task_embed, batch_size = 16, return_all_timesteps = False, guidance_weight=0):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size[0], image_size[1]), x_cond, task_embed,  return_all_timesteps = return_all_timesteps, guidance_weight=guidance_weight)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5): # 接受x1, x2, 加噪后再去噪
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, x_cond, task_embed, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step

        model_out = self.model(torch.cat([x, x_cond], dim=1), t, task_embed)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, img_cond, task_embed):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}, got({h}, {w})'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, img_cond, task_embed)

class SimpleImplicitModel(nn.Module):
    def __init__(self, hidden_dim = 512, t = 8, image_channels = 3, text_embed_dim = 512):
        super(SimpleImplicitModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.t = t

        # Convolutional layers for image processing
        self.image_conv = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

         # MLP for image feature extraction
        self.image_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # attn layer for text embedding processing
        self.task_attnpool = nn.Sequential(
                PerceiverResampler(dim = text_embed_dim, depth=2),
                nn.Linear(text_embed_dim, hidden_dim),
            )

        # Final linear layer to combine both embeddings and map to latent representation
        self.final_linear = nn.sequential(
            nn.Linear(2*hidden_dim, t*hidden_dim),
            nn.ReLU(),
            nn.Linear(t*hidden_dim, t*hidden_dim),
            nn.ReLU(),
            nn.Linear(t*hidden_dim, t*hidden_dim),
        )
    def forward(self, x_cond, text_embed):
        batch_size = x_cond.size(0)

        #process the image
        img_features = self.image_conv(x_cond) # (batch, hidden_dim, 1, 1)
        img_features = img_features.view(batch_size, -1) # Flatten (batch, hidden_dim)
        img_features = self.image_mlp(img_features) # (batch, hidden_dim)

        # Process the text embedding
        text_features = self.task_attnpool(text_embed).mean(dim=1) # (batch, hidden_dim)

        combined_features = torch.cat((img_features, text_features), dim=1) # (batch, 2*hidden_dim)

         # Final linear mapping and activation
        latent_rep = self.final_linear(combined_features) # (batch, t*hidden_dim)
        
        # Reshape to (batch, t, hidden_dim)
        latent_rep = latent_rep.view(batch_size, self.t, self.hidden_dim)
        
        return latent_rep

class ActionDecoder(nn.Module):
    def __init__(self, action_dim, hidden_dim):
        super(ActionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.ReLU(),
            nn.Linear(action_dim, action_dim),
            nn.ReLU(),
            nn.Linear(action_dim, action_dim)
        )
        
    def forward(self, z):
        action_predict = self.decoder(z)
        '''
        if self.loss_type == 'l1':
            loss = F.l1_loss(action_predict, action)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(action_predict, action)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(action_predict, action)
        else:
            raise NotImplementedError()
        '''
        return action_predict

class PretrainDecoder(nn.Module):
    def __init__(self, dir = "results", device = 'cuda'):
        super(PretrainDecoder, self).__init__()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, '../../configs/config.yaml')
        with open(config_path, "r") as file:
            cfg = yaml.safe_load(file)
        dir = os.path.join(current_dir, dir)
        data = torch.load(os.path.join(dir, 'trained_vqvae.pt'), map_location=device)
        self.n_latent_dims = cfg["vqvae"]["n_latent_dims"]
        self.vq_layer = ResidualVQ(
            dim=cfg["vqvae"]["n_latent_dims"],
            num_quantizers=cfg["vqvae"]["vqvae_groups"],
            codebook_size=cfg["vqvae"]["vqvae_n_embed"],
        ).to(device)
        self.vq_layer.load_state_dict(data["vq_embedding"])
        self.vq_layer.eval()
        if cfg["vqvae"]["action_window_size"] == 1:
            self.decoder = EncoderMLP(
                input_dim=cfg["vqvae"]["n_latent_dims"], output_dim=cfg["vqvae"]["act_dim"]
            ).to(device)
        else:
            self.decoder = EncoderMLP(
                input_dim=cfg["vqvae"]["n_latent_dims"], output_dim=cfg["vqvae"]["act_dim"] * cfg["vqvae"]["action_window_size"]
            ).to(device)
        self.decoder.eval()
        self.decoder.load_state_dict(data["decoder"])
    
    def forward(self, z):
        assert z.shape[1] == self.n_latent_dims, "the dim of z must be equal to PretrainDecoder's n_latent_dims"
        z_shape = z.shape[:-1]
        z_flat = z.view(z.size(0), -1, z.size(1))
        z_flat, vq_code, vq_loss_state = self.vq_layer(z_flat)
        z_vq = z_flat.view(*z_shape, -1)
        dec_out = self.decoder(z + (z_vq - z).detach())
        return dec_out
        
class SimpleActionDecoder(nn.Module):
    def __init__(self, action_dim, hidden_dim):
        super(SimpleActionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.ReLU(),
            nn.Linear(action_dim, action_dim),
        )
        
    def forward(self, z):
        action_predict = self.decoder(z)
        '''
        if self.loss_type == 'l1':
            loss = F.l1_loss(action_predict, action)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(action_predict, action)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(action_predict, action)
        else:
            raise NotImplementedError()
        '''
        return action_predict            
class ConditionModel(nn.Module):
    def __init__(self):
        super(ConditionModel, self).__init__()
    
    def forward(self, x):
        return x

class Preprocess(nn.Module):
    def __init__(self, hidden_dim = 16, act_len = 7, n_latent_dims = 512):
        super(Preprocess, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim * act_len, n_latent_dims),
            nn.ReLU(),
            nn.Linear(n_latent_dims, n_latent_dims),
        )
        self.batch_norm = torch.nn.BatchNorm1d(
            num_features = n_latent_dims
        )
    
    def forward(self, x):
        x = einops.rearrange(x, "N T A -> N (T A)")
        x = self.layer(x)
        x = self.batch_norm(x)
        return x

class DiffusionActionModel(nn.Module):
    def __init__(
        self,
        goal_gaussian_diffusion,
        implicit_model,
        action_decoder,
        condition_model,
        preprocess,
        action_rate = 0.5,
        action_loss_type = "l2",
        action_features = 16,
        action_scale = 10.0,
    ):
        super().__init__()
        self.action_features = action_features
        self.goal_gaussian_diffusion = goal_gaussian_diffusion
        self.implicit_model = implicit_model
        self.action_decoder = action_decoder
        self.condition_model = condition_model
        self.action_rate = action_rate
        self.action_loss_type = action_loss_type
        self.batch_norm = torch.nn.BatchNorm1d(
            num_features = action_features
        )
        self.action_scale = action_scale
        self.preprocess = preprocess



    def forward(self, img, img_cond, task_embed, action):
        z = self.implicit_model(img_cond.permute(0,2,3,1), task_embed) # (batch, (f+1)=8, hidden_dim), action大约36维, 或者直接生成（batch，512）
        z = self.batch_norm(z.permute(0,2,1))
        z = z.permute(0,2,1)
        state = self.preprocess(z)
        action_predict = self.action_decoder(state) 
        # print("action_predict:",action_predict)
        # print("action:", action)
        action = action / self.action_scale
        action = einops.rearrange(action, "N T A -> N (T A)")
        # if self.action_loss_type == 'l1':
        #     loss1 = F.l1_loss(action_predict, action)
        # elif self.action_loss_type == 'l2':
        #     loss1 = F.mse_loss(action_predict, action)
        # elif self.action_loss_type == "huber":
        #     loss1 = F.smooth_l1_loss(action_predict, action)
        # else:
        #     raise NotImplementedError()
        loss1 = (action - action_predict).abs().mean()
        condition = self.condition_model(z)
        loss2 = self.goal_gaussian_diffusion(img, img_cond, condition)
        loss = self.action_rate*loss1 + (1 - self.action_rate)*loss2
        return loss, loss1, loss2
    
    @torch.no_grad()
    def sample(self, x_cond, task_embed, batch_size = 16, return_all_timesteps = False, guidance_weight=0):
        z = self.implicit_model(x_cond.permute(0,2,3,1), task_embed)
        condition = self.condition_model(z)
        return self.goal_gaussian_diffusion.sample(x_cond, condition, batch_size, return_all_timesteps, guidance_weight)

class DiffusionActionModelWithGPT(nn.Module):
    def __init__(
        self,
        goal_gaussian_diffusion,
        action_decoder,
        condition_model,
        action_rate = 0.5,
        action_loss_type = "l2",
        action_features = 512,
        action_scale = 10.0,
        n_layer = 12,
        n_head = 4
    ):
        super().__init__()
        self.action_features = action_features
        self.goal_gaussian_diffusion = goal_gaussian_diffusion
        self.action_decoder = action_decoder
        self.condition_model = condition_model
        self.action_rate = action_rate
        self.action_loss_type = action_loss_type
        self.batch_norm = torch.nn.BatchNorm1d(
            num_features = action_features
        )
        self.action_scale = action_scale
        self.img_encoder = Encoder(
            hidden_size = action_features,
            activation_function = "relu",
            ch = 3,
            robot = False
        )
        self.gpt_config = GPT2Config(
            vocab_size = 1,
            n_embd = action_features,
            n_layer = n_layer,
            n_head = n_head
        )
        self.gpt_model = GPT2Model(self.gpt_config)



    def forward(self, img, img_cond, task_embed, action):
        
        assert  self.action_features == task_embed.shape[-1], "action_features must be equal to task_embed"
        img_input = self.img_encoder(img_cond)
        if len(img_input.shape)<3:
            img_input = torch.unsqueeze(img_input, dim = 1)
        stacked_input = torch.cat([task_embed, img_input], dim = 1)
        output = self.gpt_model(inputs_embeds = stacked_input)
        z = output["last_hidden_state"][:, -1, :] #先只用最后一个hidden，先不考虑循环输出，z[batch, hidden_dim]
        z = self.batch_norm(z)
        action_predict = self.action_decoder(z) 
        # print("action_predict:",action_predict)
        # print("action:", action)
        action = action / self.action_scale
        action = einops.rearrange(action, "N T A -> N (T A)")
        # if self.action_loss_type == 'l1':
        #     loss1 = F.l1_loss(action_predict, action)
        # elif self.action_loss_type == 'l2':
        #     loss1 = F.mse_loss(action_predict, action)
        # elif self.action_loss_type == "huber":
        #     loss1 = F.smooth_l1_loss(action_predict, action)
        # else:
        #     raise NotImplementedError()
        loss1 = (action - action_predict).abs().mean()
        condition = self.condition_model(z)
        loss2 = self.goal_gaussian_diffusion(img, img_cond, condition)
        loss = self.action_rate*loss1 + (1 - self.action_rate)*loss2
        return loss, loss1, loss2
    
    @torch.no_grad()
    def sample(self, x_cond, task_embed, batch_size = 16, return_all_timesteps = False, guidance_weight=0):
        assert  self.action_features == task_embed.shape[-1], "action_features must be equal to task_embed"
        img_input = self.img_encoder(x_cond)
        if len(img_input.shape)<3:
            img_input = torch.unsqueeze(img_input, dim = 1)
        stacked_input = torch.cat([task_embed, img_input], dim = 1)
        output = self.gpt_model(inputs_embeds = stacked_input)
        z = output["last_hidden_state"][:, -1, :] #先只用最后一个hidden，先不考虑循环输出，z[batch, hidden_dim]
        self.batch_norm.eval()
        z = self.batch_norm(z)
        condition = self.condition_model(z)
        return self.goal_gaussian_diffusion.sample(x_cond, condition, batch_size, return_all_timesteps, guidance_weight)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_action_model,
        tokenizer, 
        text_encoder, 
        train_set,
        valid_set,
        channels = 3,
        *,
        train_batch_size = 1,
        valid_batch_size = 1,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 10000,
        num_samples = 3,
        results_folder = './results',
        amp = True,
        fp16 = True,
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048, 
        cond_drop_chance=0.1,
    ):
        super().__init__()

        self.resume_training = False

        self.cond_drop_chance = cond_drop_chance

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.train_lr = train_lr
        self.adam_betas = adam_betas
        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_action_model

        self.channels = channels

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_action_model.goal_gaussian_diffusion.image_size

        # dataset and dataloader

        
        valid_ind = [i for i in range(len(valid_set))][:num_samples]

        train_set = train_set
        valid_set = Subset(valid_set, valid_ind)

        self.ds = train_set
        self.valid_ds = valid_set
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 4)
        # dl = dataloader
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.valid_dl = DataLoader(self.valid_ds, batch_size = valid_batch_size, shuffle = False, pin_memory = True, num_workers = 4)


        # optimizer

        self.opt = Adam(filter(lambda p: p.requires_grad, diffusion_action_model.parameters()), lr = train_lr, betas = adam_betas, eps = 1e-4)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_action_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        # step counter state

        self.step = 0

        # Wrap the model with DDP and set find_unused_parameters=True
        
        if self.accelerator.distributed_type == "MULTI_GPU":
            self.model = self.model.to("cuda")
            self.model = DDP(self.model, device_ids=[self.accelerator.local_process_index], find_unused_parameters=True)

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.text_encoder = \
            self.accelerator.prepare(self.model, self.opt, self.text_encoder)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        
    def load_resume(self, milestone):
        self.resume_training = True
        accelerator = self.accelerator
        device = accelerator.device

        new_model_dict = self.opt.state_dict()

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        # print(data['opt']['param_groups'])
        # print(new_model_dict['param_groups'])
        # print(data['model'].keys())

        self.step = data['step']
        # self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    #     return fid_value
    def encode_batch_text(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        # print(batch_text_embed)
        return batch_text_embed

    def sample(self, x_conds, batch_text, batch_size=1, guidance_weight=0):
        device = self.device
        task_embeds = self.encode_batch_text(batch_text)
        return self.ema.ema_model.sample(x_conds.to(device), task_embeds.to(device), batch_size=batch_size, guidance_weight=guidance_weight)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            loss_list = []
            loss_mean_list = []

            loss1_list = []
            loss1_mean_list = []

            loss2_list = []
            loss2_mean_list = []

            while self.step < self.train_num_steps:

                total_loss = 0.
                total_loss1 = 0.
                total_loss2 = 0.

                for _ in range(self.gradient_accumulate_every):
                    x, x_cond, goal, action = next(self.dl)
                    x, x_cond = x.to(device), x_cond.to(device)

                    goal_embed = self.encode_batch_text(goal)
                    ### zero whole goal_embed if p < self.cond_drop_chance
                    goal_embed = goal_embed * (torch.rand(goal_embed.shape[0], 1, 1, device = goal_embed.device) > self.cond_drop_chance).float()


                    with self.accelerator.autocast():
                        loss, loss1, loss2 = self.model(x, x_cond, goal_embed, action) 
                        loss = loss / self.gradient_accumulate_every
                        loss1 = loss1 / self.gradient_accumulate_every
                        loss2 = loss2 / self.gradient_accumulate_every
                        total_loss += loss.item()
                        total_loss1 += loss1.item()
                        total_loss2 += loss2.item()

                        self.accelerator.backward(loss)
                
                model = self.accelerator.unwrap_model(self.model)
                # params = []
                # names = []
                # for name, param in model.implicit_model.named_parameters():
                #     params.append(param.grad)
                #     names.append(name)
                # print(params)

                loss_list.append(total_loss)
                loss1_list.append(total_loss1)
                loss2_list.append(total_loss2)

                if len(loss_list) > 50 and len(loss1_list) > 50 and len(loss2_list) > 50:
                    loss_mean_list.append(np.mean(loss_list[(len(loss_list)-50):len(loss_list)]))
                    loss1_mean_list.append(np.mean(loss1_list[(len(loss1_list)-50):len(loss1_list)]))
                    loss2_mean_list.append(np.mean(loss2_list[(len(loss2_list)-50):len(loss2_list)]))

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                scale = self.accelerator.scaler.get_scale()
                
                pbar.set_description(f'loss: {total_loss:.4E}, loss scale: {scale:.1E}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.valid_batch_size) # num_samples = 1, valid_batch_size = 32, batches = [1]
                            ### get val_imgs from self.valid_dl
                            x_conds = []
                            xs = []
                            task_embeds = []
                            actions = []
                            for i, (x, x_cond, label, action) in enumerate(self.valid_dl):
                                xs.append(x)
                                x_conds.append(x_cond.to(device))
                                task_embeds.append(self.encode_batch_text(label))
                                actions.append(action)
                            
                            with self.accelerator.autocast():
                                all_xs_list = list(map(lambda n, c, e: self.ema.ema_model.sample(batch_size=n, x_cond=c, task_embed=e), batches, x_conds, task_embeds))
                        
                        print_gpu_utilization()
                        
                        gt_xs = torch.cat(xs, dim = 0) # [batch_size, 3*n, 120, 160]
                        # make it [batchsize*n, 3, 120, 160]
                        n_rows = gt_xs.shape[1] // 3
                        gt_xs = rearrange(gt_xs, 'b (n c) h w -> b n c h w', n=n_rows)
                        ### save images
                        x_conds = torch.cat(x_conds, dim = 0).detach().cpu()
                        # x_conds = rearrange(x_conds, 'b (n c) h w -> b n c h w', n=1)
                        all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu()
                        all_xs = rearrange(all_xs, 'b (n c) h w -> b n c h w', n=n_rows)

                        gt_first = gt_xs[:, :1]
                        gt_last = gt_xs[:, -1:]



                        if self.step == self.save_and_sample_every:
                            os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
                            gt_img = torch.cat([gt_first, gt_last, gt_xs], dim=1)
                            gt_img = rearrange(gt_img, 'b n c h w -> (b n) c h w', n=n_rows+2)
                            utils.save_image(gt_img, str(self.results_folder / f'imgs/gt_img.png'), nrow=n_rows+2)

                        os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
                        pred_img = torch.cat([gt_first, gt_last,  all_xs], dim=1)
                        pred_img = rearrange(pred_img, 'b n c h w -> (b n) c h w', n=n_rows+2)
                        utils.save_image(pred_img, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow=n_rows+2)

                        if self.resume_training:
                            loss_img_path = os.path.join(self.results_folder, "imgs/loss-stage2/")
                        else:
                            loss_img_path = os.path.join(self.results_folder, "imgs/loss/")
                        
                        if not os.path.exists(loss_img_path):
                            os.makedirs(loss_img_path)

                        self.save(milestone)
                        plt.figure()
                        axis_x = range(len(loss_mean_list))
                        plt.plot(axis_x,loss_mean_list)
                        plt.savefig(os.path.join(loss_img_path, f'loss-step{self.step // self.save_and_sample_every}.png'))

                        plt.figure()
                        axis_x = range(len(loss1_mean_list))
                        plt.plot(axis_x,loss1_mean_list)
                        plt.savefig(os.path.join(loss_img_path, f'loss1-step{self.step // self.save_and_sample_every}.png'))

                        plt.figure()
                        axis_x = range(len(loss2_mean_list))
                        plt.plot(axis_x,loss2_mean_list)
                        plt.savefig(os.path.join(loss_img_path, f'loss2-step{self.step // self.save_and_sample_every}.png'))


                pbar.update(1)

        accelerator.print('training complete')
