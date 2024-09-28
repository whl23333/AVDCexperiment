from .goal_diffusion import GoalGaussianDiffusion, Trainer
from .goal_diffusion_v1 import GoalGaussianDiffusion as GoalGaussianDiffusion_v1, Trainer as Trainer_v1
from .goal_diffusion_policy import GoalGaussianDiffusion as GoalGaussianDiffusionPolicy, Trainer as TrainerPolicy
from .diffusion_policy_baseline.unet import Unet1D, TransformerNet
from .unet import UnetMW as Unet
from .unet import UnetMWFlow as Unet_flow
from .unet import UnetThor as Unet_thor
from .unet import UnetBridge as Unet_bridge
from .unet import UnetMWChange as UnetChange
from .unet import NewUnetMW
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms as T
from einops import rearrange
import torch
from PIL import Image
from torch import nn
import numpy as np
from .goal_diffusion_change import GoalGaussianDiffusion as GoalGaussianDiffusionChange
from .goal_diffusion_change import Trainer as TrainerChange
from .goal_diffusion_change import ActionDecoder, ConditionModel, Preprocess, DiffusionActionModel, SimpleActionDecoder, PretrainDecoder
from .goal_diffusion_change import DiffusionActionModelWithGPT
import os
import yaml
from .ImgTextPerceiver import ImgTextPerceiverModel, ConvImgTextPerceiverModel, TwoStagePerceiverModel

def get_diffusion_policy_T(ckpt_dir='../ckpts/diffusion_policy_T', milestone=1, sampling_timesteps=10):
    unet = TransformerNet()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusionPolicy(
        channels=4,
        model=unet,
        image_size=10,
        timesteps=100,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    trainer = TrainerPolicy(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[0],
        valid_set=[0],
        train_lr=1e-4,
        train_num_steps =100000,
        save_and_sample_every =2500,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =32,
        valid_batch_size =1,
        gradient_accumulate_every = 1,
        num_samples=1, 
        results_folder =ckpt_dir,
        fp16 =True,
        amp=True,
    )

    trainer.load(milestone)
    return trainer

class DiffusionPolicy_T():
    def __init__(self, milestone=10, amp=True, sampling_timesteps=10):
        self.policy = get_diffusion_policy_T(milestone=milestone, sampling_timesteps=sampling_timesteps)
        self.amp = amp
        self.transform = T.Compose([
            T.Resize((320, 240)),
            T.CenterCrop((128, 128)),
            T.ToTensor(),
        ])

    def __call__(self,
                obs: np.array,
                task: str,
            ):
        device = self.policy.device
        obs = torch.stack([self.transform(Image.fromarray(o)) for o in obs], dim=0).float().to(device).unsqueeze(0)
        with torch.no_grad():
            return self.policy.sample(obs, [task]).cpu().squeeze(0).numpy()

def get_diffusion_policy(ckpt_dir='../ckpts/diffusion_policy', milestone=1, sampling_timesteps=10):
    unet = Unet1D()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusionPolicy(
        channels=4,
        model=unet,
        image_size=16,
        timesteps=100,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    trainer = TrainerPolicy(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[0],
        valid_set=[0],
        train_lr=1e-4,
        train_num_steps =100000,
        save_and_sample_every =2500,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =32,
        valid_batch_size =1,
        gradient_accumulate_every = 1,
        num_samples=1, 
        results_folder =ckpt_dir,
        fp16 =True,
        amp=True,
    )

    trainer.load(milestone)
    return trainer

class DiffusionPolicy():
    def __init__(self, milestone=10, amp=True, sampling_timesteps=10):
        self.policy = get_diffusion_policy(milestone=milestone, sampling_timesteps=sampling_timesteps)
        self.amp = amp
        self.transform = T.Compose([
            T.Resize((320, 240)),
            T.CenterCrop((128, 128)),
            T.ToTensor(),
        ])

    def __call__(self,
                obs: np.array,
                task: str,
            ):
        device = self.policy.device
        obs = torch.stack([self.transform(Image.fromarray(o)) for o in obs], dim=0).float().to(device).unsqueeze(0)
        with torch.no_grad():
            return self.policy.sample(obs, [task]).cpu().squeeze(0).numpy()


def get_video_model(ckpts_dir='../ckpts/metaworld', milestone=24, flow=False, timestep=100):
    unet = Unet_flow() if flow else Unet()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = 8
    target_size = (128, 128)
    channels = 3 if not flow else 2

    diffusion = GoalGaussianDiffusion(
        channels=channels*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=timestep,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    
    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load(milestone)
    return trainer
        
def get_video_model_change(ckpts_dir='../ckpts/metaworld', milestone=24, flow=False, timestep=100):
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, '../../configs/config.yaml')
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    
    unet = Unet_flow() if flow else UnetChange()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = cfg["sample_per_seq"]
    target_size = (128, 128)
    channels = 3 if not flow else 2

    diffusion = GoalGaussianDiffusionChange(
        channels=channels*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=timestep,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    # implicit_model
    model_name = cfg["models"]["implicit_model"]["model_name"]
    model_params = cfg["models"]["implicit_model"]["params"]
    class_ = globals()[model_name]
    implicit_model = class_(**model_params)

    # action_decoder
    model_name = cfg["models"]["action_decoder"]["model_name"]
    model_params = cfg["models"]["action_decoder"]["params"]
    class_ = globals()[model_name]
    action_decoder = class_(**model_params)

    # condition_model
    condition_model = ConditionModel()

    # preprocess
    model_name = cfg["models"]["preprocess"]["model_name"]
    model_params = cfg["models"]["preprocess"]["params"]
    class_ = globals()[model_name]
    preprocess = class_(**model_params)

    diffusion_action_model11 = DiffusionActionModel(
        diffusion,
        implicit_model,
        action_decoder,
        condition_model,
        preprocess,
        action_rate = cfg["models"]["diffusion_action_model"]["params"]["action_rate"],
    )
    
    trainer = TrainerChange(
        diffusion_action_model=diffusion_action_model11,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load_resume(milestone)
    return trainer   


def get_video_model_GPT(ckpts_dir='../ckpts/metaworld', milestone=24, flow=False, timestep=100):
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, '../../configs/config.yaml')
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    
    unet = Unet_flow() if flow else NewUnetMW()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = cfg["sample_per_seq"]
    target_size = (128, 128)
    channels = 3 if not flow else 2

    diffusion = GoalGaussianDiffusionChange(
        channels=channels*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=timestep,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    # implicit_model
    model_name = cfg["models"]["implicit_model"]["model_name"]
    model_params = cfg["models"]["implicit_model"]["params"]
    class_ = globals()[model_name]
    implicit_model = class_(**model_params)

    # action_decoder
    model_name = cfg["models"]["action_decoder"]["model_name"]
    model_params = cfg["models"]["action_decoder"]["params"]
    class_ = globals()[model_name]
    action_decoder = class_(**model_params)

    # condition_model
    condition_model = ConditionModel()

    # preprocess
    model_name = cfg["models"]["preprocess"]["model_name"]
    model_params = cfg["models"]["preprocess"]["params"]
    class_ = globals()[model_name]
    preprocess = class_(**model_params)

    diffusion_action_model_GPT = DiffusionActionModelWithGPT(
        diffusion,
        action_decoder,
        condition_model,
        action_rate = cfg["models"]["diffusion_action_model"]["params"]["action_rate"],
        n_layer=12,
        n_head=4
    )
    
    trainer = TrainerChange(
        diffusion_action_model=diffusion_action_model_GPT,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load_resume(milestone)
    return trainer        


def get_video_model_thor(ckpts_dir='../ckpts/ithor', milestone=30):
    unet = Unet_thor()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = 8
    target_size = (64, 64)
    channels = 3

    diffusion = GoalGaussianDiffusion(
        channels=channels*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=100,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    
    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load(milestone)
    return trainer

def get_video_model_bridge(ckpts_dir='../ckpts/bridge', milestone=42):
    unet = Unet_bridge()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = 8
    target_size = (48, 64)
    channels = 3

    diffusion = GoalGaussianDiffusion_v1(
        model=unet,
        image_size=target_size,
        channels=channels*(sample_per_seq-1),
        timesteps=100,
        sampling_timesteps=100,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    
    trainer = Trainer_v1(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load(milestone)
    return trainer

def pred_video(model, frame_0, task, flow=False):
    device = model.device
    original_shape = frame_0.shape
    center = (original_shape[1]//2, original_shape[0]//2)
    xpad, ypad = center[0]-64, center[1]-64

    channels = 3 if not flow else 2
    
    transform = T.Compose([
        T.CenterCrop((128, 128)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]
    preds = rearrange(model.sample(image.to(device), text).cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    if not flow:
        preds = torch.cat([image, preds], dim=0)
    # pad the image back to original shape (both sides)
    images = torch.nn.functional.pad(preds, (xpad, xpad, ypad, ypad))
    return images.numpy().transpose(0, 2, 3, 1) * 128 if flow else (images.numpy()*255).astype('uint8')

def pred_video_thor(model, frame_0, task):
    channels=3
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]
    preds = rearrange(model.sample(image, text).cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    preds = torch.cat([image, preds], dim=0)
    return (preds.numpy()*255).astype('uint8')

def pred_video_bridge(model, frame_0, task):
    channels=3
    transform = T.Compose([
        T.Resize((48, 64)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]
    preds = rearrange(model.sample(image, text).cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    preds = torch.cat([image, preds], dim=0)
    return (preds.numpy()*255).astype('uint8')
