# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import glob
import json
import math
import os
import types
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from torch import nn

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    xFuserLongContextAttention)
from ..dist.wan_xfuser import usp_attn_forward_framepack
from .cache_utils import TeaCache
from .wan_camera_adapter import SimpleAdapter

from .wan_transformer3d import flash_attention, attention
from .wan_transformer3d import sinusoidal_embedding_1d
from .wan_transformer3d import get_resize_crop_region_for_grid
from .wan_transformer3d import rope_params, get_1d_rotary_pos_embed_riflex
from .wan_transformer3d import WanRMSNorm, WanLayerNorm
from .wan_transformer3d import Head, MLPProj

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
    print("Using Flash Attention 3 !!!")
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    major, minor = torch.cuda.get_device_capability(0)
    if f"{major}.{minor}" == "8.0":
        from sageattention_sm80 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "8.6":
        from sageattention_sm86 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "8.9":
        from sageattention_sm89 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif major>=9:
        from sageattention_sm90 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
except:
    try:
        from sageattention import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    except:
        sageattn = None
        SAGE_ATTENTION_AVAILABLE = False


def apply_rotary_emb_transposed(x, freqs):
    cos, sin = freqs.unsqueeze(-2).chunk(2, dim=-1)
    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    out = x.float() * cos + x_rotated.float() * sin
    out = out.to(x)
    return out

class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = attention(
            q=apply_rotary_emb_transposed(q, freqs).to(dtype),
            k=apply_rotary_emb_transposed(k, freqs).to(dtype),
            v=v.to(dtype),
            k_lens=seq_lens,
            window_size=self.window_size)
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, dtype):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)

        # compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, dtype):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img.to(dtype))).view(b, -1, n, d)
        v_img = self.v_img(context_img.to(dtype)).view(b, -1, n, d)

        img_x = attention(
            q.to(dtype), 
            k_img.to(dtype), 
            v_img.to(dtype), 
            k_lens=None
        )
        img_x = img_x.to(dtype)
        # compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.float32
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype)
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # cross-attention
            x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype)

            # ffn function
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)
            
            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class VideoRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim, theta):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta

    @torch.no_grad()
    def get_frequency(self, dim, pos):
        T, H, W = pos.shape
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device)[: (dim // 2)] / dim))
        freqs = torch.outer(freqs, pos.reshape(-1)).unflatten(-1, (T, H, W)).repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    def forward_inner(self, frame_indices, height, width, device):
        GT, GY, GX = torch.meshgrid(
            frame_indices.to(device=device, dtype=torch.float32),
            torch.arange(0, height, device=device, dtype=torch.float32),
            torch.arange(0, width, device=device, dtype=torch.float32),
            indexing="ij"
        )

        FCT, FST = self.get_frequency(self.DT, GT)
        FCY, FSY = self.get_frequency(self.DY, GY)
        FCX, FSX = self.get_frequency(self.DX, GX)
        result = torch.cat([FCT, FCY, FCX, FST, FSY, FSX], dim=0)

        return result.to(device)

    @torch.no_grad()
    def forward(self, frame_indices, height, width, device):
        frame_indices = frame_indices.unbind(0)
        results = [self.forward_inner(f, height, width, device) for f in frame_indices]
        results = torch.stack(results, dim=0)
        return results

class VideoPatchEmbedForCleanLatents(nn.Module):
    def __init__(self, inner_dim):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    @torch.no_grad()
    def initialize_weight_from_another_conv3d(self, another_layer):
        weight = another_layer.weight.detach().clone()
        bias = another_layer.bias.detach().clone()

        sd = {
            'proj.weight': weight.clone(),
            'proj.bias': bias.clone(),
            'proj_2x.weight': einops.repeat(weight, 'b c t h w -> b c (t tk) (h hk) (w wk)', tk=2, hk=2, wk=2) / 8.0,
            'proj_2x.bias': bias.clone(),
            'proj_4x.weight': einops.repeat(weight, 'b c t h w -> b c (t tk) (h hk) (w wk)', tk=4, hk=4, wk=4) / 64.0,
            'proj_4x.bias': bias.clone(),
        }

        sd = {k: v.clone() for k, v in sd.items()}

        self.load_state_dict(sd)
        return

def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')

def center_down_sample_3d(x, kernel_size):
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)

class WanTransformer3DModel_FramePack(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        add_ref_conv=False,
        in_dim_ref_conv=16,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.use_gradient_checkpointing = False
        self.clean_x_embedder = VideoPatchEmbedForCleanLatents(dim)
        self.patch_embedding_new = nn.Conv3d(16, dim, kernel_size=patch_size, stride=patch_size)

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim
        self.rope = VideoRotaryPosEmbed((d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)), theta=10000)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        self.teacache = None
        self.gradient_checkpointing = False
        self.sp_world_size = 1
        self.sp_world_rank = 0
    
    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True
    ):
        self.teacache = TeaCache(
            coefficients, num_steps, rel_l1_thresh=rel_l1_thresh, num_skip_start_steps=num_skip_start_steps, offload=offload
        )

    def disable_teacache(self):
        self.teacache = None

    def enable_multi_gpus_inference(self,):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward_framepack, block.self_attn)

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value
    
    def gradient_checkpointing_method(self, block, *args):
        if self.use_gradient_checkpointing:
            result = torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)
        else:
            result = block(*args)
        return result

    def process_input_hidden_states(
            self,
            latents, latent_indices=None,
            clean_latents=None, clean_latent_indices=None,
            clean_latents_2x=None, clean_latent_2x_indices=None,
            clean_latents_4x=None, clean_latent_4x_indices=None):
        # compress feature map (c,T,H,W) into (C,T,H//2,W//2)
        # latents: [1, 16, 9, 80, 76] -> hidden_states [1, 3072, 9, 40, 38]
        # latent_indices: [28, 29, 30, 31, 32, 33, 34, 35, 36]

        hidden_states = self.gradient_checkpointing_method(self.patch_embedding_new, latents)
        B, C, T, H, W = hidden_states.shape

        if latent_indices is None:
            latent_indices = torch.arange(0, T).unsqueeze(0).expand(B, -1)
        
        grid_sizes = torch.stack([torch.tensor(u.shape[1:], dtype=torch.long) for u in hidden_states]) 
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        

        rope_freqs = self.rope(frame_indices=latent_indices, height=H, width=W, device=hidden_states.device) #[1, 256, 9, 40, 38]
        rope_freqs = rope_freqs.flatten(2).transpose(1, 2) # [1, 13680, 256]

        if clean_latents is not None and clean_latent_indices is not None:
            # compress clean_latents (T,H,W) into (T,H//2,W//2)
            # clean_latent_indices: [ 0, 37]
            clean_latents = clean_latents.to(hidden_states)
            clean_latents = self.gradient_checkpointing_method(self.clean_x_embedder.proj, clean_latents)
            clean_latents = clean_latents.flatten(2).transpose(1, 2)

            clean_latent_rope_freqs = self.rope(frame_indices=clean_latent_indices, height=H, width=W, device=clean_latents.device)
            clean_latent_rope_freqs = clean_latent_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_rope_freqs, rope_freqs], dim=1)

        if clean_latents_2x is not None and clean_latent_2x_indices is not None:
            # compress clean_latents_2x (T,H,W) into (T//2,H//4,W//4)
            # clean_latent_2x_indices: [38, 39]
            clean_latents_2x = clean_latents_2x.to(hidden_states)
            clean_latents_2x = pad_for_3d_conv(clean_latents_2x, (2, 4, 4))
            clean_latents_2x = self.gradient_checkpointing_method(self.clean_x_embedder.proj_2x, clean_latents_2x)
            clean_latents_2x = clean_latents_2x.flatten(2).transpose(1, 2)

            clean_latent_2x_rope_freqs = self.rope(frame_indices=clean_latent_2x_indices, height=H, width=W, device=clean_latents_2x.device)
            clean_latent_2x_rope_freqs = pad_for_3d_conv(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = center_down_sample_3d(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = clean_latent_2x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents_2x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_2x_rope_freqs, rope_freqs], dim=1)

        if clean_latents_4x is not None and clean_latent_4x_indices is not None:
            # compress clean_latents_2x (T,H,W) into (T//4,H//8,W//8)
            # clean_latent_4x_indices: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
            clean_latents_4x = clean_latents_4x.to(hidden_states)
            clean_latents_4x = pad_for_3d_conv(clean_latents_4x, (4, 8, 8))
            clean_latents_4x = self.gradient_checkpointing_method(self.clean_x_embedder.proj_4x, clean_latents_4x)
            clean_latents_4x = clean_latents_4x.flatten(2).transpose(1, 2)

            clean_latent_4x_rope_freqs = self.rope(frame_indices=clean_latent_4x_indices, height=H, width=W, device=clean_latents_4x.device)
            clean_latent_4x_rope_freqs = pad_for_3d_conv(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = center_down_sample_3d(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = clean_latent_4x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents_4x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_4x_rope_freqs, rope_freqs], dim=1)

        return hidden_states, rope_freqs, grid_sizes

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        cond_flag=True,
        # For FramePack
        latent_indices=None,
        clean_latents=None, clean_latent_indices=None,
        clean_latents_2x=None, clean_latent_2x_indices=None,
        clean_latents_4x=None, clean_latent_4x_indices=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        x, rope_freqs, grid_sizes = self.process_input_hidden_states(x, latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices)
        
        # add control adapter
        if self.control_adapter is not None and y_camera is not None:
            y_camera = self.control_adapter(y_camera)
            x = [u + v for u, v in zip(x, y_camera)]
        
        if self.ref_conv is not None and full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += full_ref.size(1)
            x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref, x)]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            rope_freqs = torch.chunk(rope_freqs, self.sp_world_size, dim=1)[self.sp_world_rank]
        
        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = should_calc
            else:
                should_calc = self.teacache.should_calc
        
        # TeaCache
        if self.teacache is not None:
            if not should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            rope_freqs,
                            context,
                            context_lens,
                            dtype,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=rope_freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype
                        )
                        x = block(x, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        rope_freqs,
                        context,
                        context_lens,
                        dtype,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=rope_freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype
                    )
                    x = block(x, **kwargs)

        if self.sp_world_size > 1:
            x = get_sp_group().all_gather(x, dim=1)

        if self.ref_conv is not None and full_ref is not None:
            full_ref_length = full_ref.size(1)
            x = x[:, full_ref_length:]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x


    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                from diffusers.models.modeling_utils import \
                    load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(state_dict)
                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )
        
        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
            model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight']
            model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
            state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']
        
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        model = model.to(torch_dtype)
        return model