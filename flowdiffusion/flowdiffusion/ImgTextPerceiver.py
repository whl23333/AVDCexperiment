from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# 对原有perceiver的更改，每一次循环中，先后对（latent，img），（latent，text）做cross attention
class ImgTextPerceiverModel(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        img_input_channels = 3,
        img_input_axis = 2,
        text_input_channels = 512,
        text_input_axis = 1,
        num_latents = 8,
        latent_dim = 64,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.img_input_axis = img_input_axis
        self.text_input_axis = text_input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.fourier_encode_data = fourier_encode_data

        # img fourier_encode_data
        img_fourier_channels = (img_input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        img_input_dim = img_fourier_channels + img_input_channels

        # text fourier_encode_data
        text_fourier_channels = (text_input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        text_input_dim = text_fourier_channels + text_input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        img_get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, img_input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = img_input_dim)
        text_get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, text_input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = text_input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        img_get_cross_attn, text_get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (img_get_cross_attn, text_get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                img_get_cross_attn(**cache_args),
                text_get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(
        self,
        img_data,
        text_data,
        mask = None,
        return_embeddings = True
    ):
        b, *img_axis, _, img_device, img_dtype = *img_data.shape, img_data.device, img_data.dtype
        assert len(img_axis) == self.img_input_axis, 'input img data must have the right number of axis'

        b, *text_axis, _, text_device, text_dtype = *text_data.shape, text_data.device, text_data.dtype
        assert len(text_axis) == self.text_input_axis, 'input text data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            img_axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=img_device, dtype=img_dtype), img_axis))
            img_pos = torch.stack(torch.meshgrid(*img_axis_pos, indexing = 'ij'), dim = -1)
            img_enc_pos = fourier_encode(img_pos, self.max_freq, self.num_freq_bands)
            img_enc_pos = rearrange(img_enc_pos, '... n d -> ... (n d)')
            img_enc_pos = repeat(img_enc_pos, '... -> b ...', b = b)

            img_data = torch.cat((img_data, img_enc_pos), dim = -1)

            text_axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=text_device, dtype=text_dtype), text_axis))
            text_pos = torch.stack(torch.meshgrid(*text_axis_pos, indexing = 'ij'), dim = -1)
            text_enc_pos = fourier_encode(text_pos, self.max_freq, self.num_freq_bands)
            text_enc_pos = rearrange(text_enc_pos, '... n d -> ... (n d)')
            text_enc_pos = repeat(text_enc_pos, '... -> b ...', b = b)

            text_data = torch.cat((text_data, text_enc_pos), dim = -1)

        # concat to channels of data and flatten axis

        img_data = rearrange(img_data, 'b ... d -> b (...) d')
        text_data = rearrange(text_data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)

        # layers

        for img_cross_attn, text_cross_attn, cross_ff, self_attns in self.layers:
            x = img_cross_attn(x, context = img_data, mask = mask) + x
            x = text_cross_attn(x, context = text_data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)

class ConvImgTextPerceiverModel(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        first_img_channels = 3,
        img_input_channels = 64,
        img_input_axis = 2,
        text_input_channels = 512,
        text_input_axis = 1,
        num_latents = 8,
        latent_dim = 64,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.img_input_axis = img_input_axis
        self.text_input_axis = text_input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.fourier_encode_data = fourier_encode_data
        self.first_img_channels = first_img_channels

        # self.conv_layer
        self.conv_layer = nn.Sequential(
            nn.Conv2d(first_img_channels, (img_input_channels//4), kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d((img_input_channels//4), (img_input_channels//2), kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d((img_input_channels//2), img_input_channels, kernel_size=3, stride=2, padding=1),
        )

        # img fourier_encode_data
        img_fourier_channels = (img_input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        img_input_dim = img_fourier_channels + img_input_channels

        # text fourier_encode_data
        text_fourier_channels = (text_input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        text_input_dim = text_fourier_channels + text_input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        img_get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, img_input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = img_input_dim)
        text_get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, text_input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = text_input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        img_get_cross_attn, text_get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (img_get_cross_attn, text_get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                img_get_cross_attn(**cache_args),
                text_get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(
        self,
        img_data,
        text_data,
        mask = None,
        return_embeddings = True
    ):
        
        img_data = self.conv_layer(img_data.permute(0,3,1,2))
        img_data = img_data.permute(0,2,3,1)
        b, *img_axis, _, img_device, img_dtype = *img_data.shape, img_data.device, img_data.dtype
        assert len(img_axis) == self.img_input_axis, 'input img data must have the right number of axis'

        b, *text_axis, _, text_device, text_dtype = *text_data.shape, text_data.device, text_data.dtype
        assert len(text_axis) == self.text_input_axis, 'input text data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            img_axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=img_device, dtype=img_dtype), img_axis))
            img_pos = torch.stack(torch.meshgrid(*img_axis_pos, indexing = 'ij'), dim = -1)
            img_enc_pos = fourier_encode(img_pos, self.max_freq, self.num_freq_bands)
            img_enc_pos = rearrange(img_enc_pos, '... n d -> ... (n d)')
            img_enc_pos = repeat(img_enc_pos, '... -> b ...', b = b)

            img_data = torch.cat((img_data, img_enc_pos), dim = -1)

            text_axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=text_device, dtype=text_dtype), text_axis))
            text_pos = torch.stack(torch.meshgrid(*text_axis_pos, indexing = 'ij'), dim = -1)
            text_enc_pos = fourier_encode(text_pos, self.max_freq, self.num_freq_bands)
            text_enc_pos = rearrange(text_enc_pos, '... n d -> ... (n d)')
            text_enc_pos = repeat(text_enc_pos, '... -> b ...', b = b)

            text_data = torch.cat((text_data, text_enc_pos), dim = -1)

        # concat to channels of data and flatten axis

        img_data = rearrange(img_data, 'b ... d -> b (...) d')
        text_data = rearrange(text_data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)

        # layers

        for img_cross_attn, text_cross_attn, cross_ff, self_attns in self.layers:
            x = img_cross_attn(x, context = img_data, mask = mask) + x
            x = text_cross_attn(x, context = text_data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        return_embeddings = False
    ):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            data = torch.cat((data, enc_pos), dim = -1)

        # concat to channels of data and flatten axis

        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)

        # layers

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)

class ReceivePerceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        # self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.num_latents = num_latents

        self.latent_dim = latent_dim

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(
        self,
        data,
        latents,
        mask = None,
        return_embeddings = False
    ):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        assert latents.shape[1] == self.num_latents, 'input latents number must be equal to self.num_latents'
        assert latents.shape[2] == self.latent_dim, 'input latents dim must be equal to self.latent_dim'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            data = torch.cat((data, enc_pos), dim = -1)

        # concat to channels of data and flatten axis

        data = rearrange(data, 'b ... d -> b (...) d')

        # x = repeat(self.latents, 'n d -> b n d', b = b)
        x = latents

        # layers

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)

    
class TwoStagePerceiverModel(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        first_img_channels = 3,
        img_input_channels = 64,
        img_input_axis = 2,
        text_input_channels = 512,
        text_input_axis = 1,
        num_latents = 8,
        latent_dim = 64,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
    ):
        super().__init__()
        self.text_perceiver = Perceiver(
            input_channels = text_input_channels,
            input_axis = text_input_axis,
            num_freq_bands = num_freq_bands,
            max_freq = max_freq,
            depth = depth,

            num_latents = num_latents,
            latent_dim = latent_dim,
            cross_heads = cross_heads,
            latent_heads = latent_heads,
            cross_dim_head = cross_dim_head,
            latent_dim_head = latent_dim_head,
            num_classes = num_classes,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            weight_tie_layers = weight_tie_layers,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = fourier_encode_data,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = self_per_cross_attn
        )

        self.conv_layer = nn.Sequential(
            nn.Conv2d(first_img_channels, (img_input_channels//4), kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d((img_input_channels//4), (img_input_channels//2), kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d((img_input_channels//2), img_input_channels, kernel_size=3, stride=2, padding=1),
        )

        self.img_perceiver = ReceivePerceiver(
            input_channels = img_input_channels,
            input_axis = img_input_axis,
            num_freq_bands = num_freq_bands,
            max_freq = max_freq,
            depth = depth,

            num_latents = num_latents,
            latent_dim = latent_dim,
            cross_heads = cross_heads,
            latent_heads = latent_heads,
            cross_dim_head = cross_dim_head,
            latent_dim_head = latent_dim_head,
            num_classes = num_classes,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            weight_tie_layers = weight_tie_layers,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = fourier_encode_data,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = self_per_cross_attn
        )
    
    def forward(
        self,
        img_data,
        text_data,
        mask = None,
        return_embeddings = True    
    ):
        z = self.text_perceiver(text_data, mask = mask, return_embeddings = return_embeddings)
        img_data = self.conv_layer(img_data.permute(0,3,1,2))
        img_data = img_data.permute(0,2,3,1)
        x = self.img_perceiver(img_data, z, mask = mask, return_embeddings = return_embeddings)

        return x
        

