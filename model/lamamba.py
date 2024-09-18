import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# triton cross scan, 2x speed than pytorch implementation =========================
try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1

try:
    from .vmamba import *
except:
    from vmamba import *

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

    


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    pos_embed = pos_embed.reshape([1, grid_size, grid_size, embed_dim])
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



#################################################################################
#                Patch Embedding/Merging/Expanding Functions                    #
#################################################################################


class PatchEmbed2D_(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 1.
        in_chans (int): Number of input image channels. Default: 4.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=1, in_chans=4, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_expand_pad = self._patch_expand_pad_channel_first if channel_first else self._patch_expand_pad_channel_last
        self.expand = Linear(dim, (dim_scale * dim), bias=False)
        self.norm = norm_layer(dim // dim_scale)


    @staticmethod
    def _patch_expand_pad_channel_last(self, x: torch.Tensor):
        H, W, C = x.shape[-3:]
        x = self.expand(x) # B H W dim_scale*C
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x = self.norm(x)

        return x

    @staticmethod
    def _patch_expand_pad_channel_first(self, x: torch.Tensor):
        C, H, W = x.shape[-3:]
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.expand(x) # B H W dim_scale*C
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')

        return x

    def forward(self, x):
        x = self._patch_expand_pad(self, x)
        return x


class FinalPatchExpanding2D(nn.Module):
    # keep the channel dim the same
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_expand_pad = self._patch_expand_pad_channel_first if channel_first else self._patch_expand_pad_channel_last
        self.expand = Linear(dim, (dim_scale**2 * dim), bias=False)
        self.norm = norm_layer(dim)


    @staticmethod
    def _patch_expand_pad_channel_last(self, x: torch.Tensor):
        H, W, C = x.shape[-3:]
        x = self.expand(x) # B H W (dim_scale**2)*C
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C)
        x = self.norm(x)

        return x

    @staticmethod
    def _patch_expand_pad_channel_first(self, x: torch.Tensor):
        C, H, W = x.shape[-3:]
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.expand(x) # B H W dim_scale*C
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C)
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')

        return x

    def forward(self, x):
        x = self._patch_expand_pad(self, x)
        return x

#################################################################################
#                            Window Local Attention                             #
#################################################################################


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

#################################################################################
#                                LaMamba-Diff                                   #
#################################################################################



class LaMambaBlock_conditioned(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        condition_dim: int = 768,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        full_condition_project=False,
        # =============================
        input_resolution=32,
        shift_size=0, # for swin attention
        window_size=8,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        fused_window_process=False, 
        # =============================
        continuous_scan=True,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        decoder=False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.decoder=decoder
        # do not project condition if dimension this avoids dim->dim at the bottleneck layers
        if full_condition_project or condition_dim != hidden_dim:
            self.condition_projection = nn.Linear(condition_dim,hidden_dim, bias=True)
        else:
            self.condition_projection = nn.Identity()

        ## windowAttention
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm_swin = norm_layer(hidden_dim)
        self.input_resolution = input_resolution
        if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            hidden_dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution, self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process
        ## windowAttention


        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                continuous_scan=continuous_scan,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

        # diffusion conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim, bias=True)
        )

    def _forward_swinAttn(self, x: torch.Tensor):
        B, H, W, C = x.size()
        # swinAttention
        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x

        return x

    def _forward(self, input: torch.Tensor, c):

        c = self.condition_projection(c)[:,None,None,:]

        shift_vss, scale_vss, gate_vss, shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=-1)


        if self.ssm_branch:
            x = input + gate_vss*self.op(modulate(self.norm(input), shift_vss, scale_vss))
        
        x = x + gate_attn*self._forward_swinAttn(modulate(self.norm_swin(x), shift_attn, scale_attn))


        if self.mlp_branch:
            x = x + gate_mlp*self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

    def forward(self, input: torch.Tensor, c):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, c)
        else:
            return self._forward(input, c)


class LaMambaDiff_layer(nn.Module):
    def __init__(
        self,
        dim,
        condition_dim=384,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=None,
        channel_first=False,
        bottleneck_stage=False,
        full_condition_project=False,
        # ===========================
        input_resolution=32,
        window_size=8,
        num_heads=3,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        fused_window_process=False, 
        # ===========================
        continuous_scan=True,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        **kwargs,):

        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
    
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = None
     

        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            block_cls = LaMambaBlock_conditioned # all conditioned
            blocks.append(block_cls(
                hidden_dim=dim, 
                condition_dim=condition_dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                full_condition_project=full_condition_project,
                # attn
                input_resolution=input_resolution,
                shift_size=0 if (d % 2 == 0) else window_size // 2,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                fused_window_process=fused_window_process,
                # attn
                continuous_scan=continuous_scan,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))
        
        self.blocks =  nn.ModuleList(blocks)



    def forward(self, x, c):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, c)
            else:
                x = blk(x, c)

        skip_feat=x

        if self.downsample is not None:
            x = self.downsample(x)

        return x, skip_feat


class LaMambaDiff_layer_up(nn.Module):
    def __init__(
        self,
        dim,
        condition_dim=384,
        upsample=None,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        channel_first=False,
        bottleneck_stage=False,
        skip_concat=False,
        full_condition_project=False,
        # ===========================
        residual_dims = 0,
        input_resolution=32,
        window_size=8,
        num_heads=3,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        fused_window_process=False, 
        # ===========================
        continuous_scan=True,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        decoder=True,
        **kwargs,):

        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
    
        if upsample is not None:
            self.upsample = upsample
        else:
            self.upsample = None

        self.skip_concat = skip_concat
        if skip_concat:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(dim*2, dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True))

        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            block_cls = LaMambaBlock_conditioned # all conditioned


            blocks.append(block_cls(
                hidden_dim=dim if d !=0 else dim + residual_dims, 
                condition_dim=condition_dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                full_condition_project=full_condition_project,
                # attn
                input_resolution=input_resolution,
                shift_size=0 if (d % 2 == 0) else window_size // 2,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                fused_window_process=fused_window_process,
                # attn
                continuous_scan=continuous_scan,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                decoder=decoder,
            ))
        
        self.blocks =  nn.ModuleList(blocks)


    def forward(self, x, c, skip_connection=None):
        if self.upsample is not None:
            x = self.upsample(x)
        if skip_connection is not None:
            if self.skip_concat:
                # concat
                x = torch.concat([x,skip_connection],dim=-1)
                B, W, H, C = x.size()
                x = self.skip_conv(x.view(B, C, W, H))
                x = x.view(B, W, H,int(C/2))
            else:
                # add
                x += skip_connection
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, c)
            else:
                x = blk(x, c)
        return x

class LaMambaDiff_bottleneck(nn.Module):
    def __init__(
        self,
        dim,
        condition_dim=384,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        channel_first=False,
        # ===========================
        input_resolution=32,
        window_size=8,
        num_heads=3,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        fused_window_process=False, 
        bottleneck_depth=1,
        # ===========================
        continuous_scan=True,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        **kwargs,):

        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
     

        blocks = []
        for d in range(bottleneck_depth):
            block_cls = LaMambaBlock_conditioned # final
            blocks.append(block_cls(
                hidden_dim=dim, 
                condition_dim=condition_dim,
                drop_path=[0.],
                norm_layer=norm_layer,
                channel_first=channel_first,
                # attn
                input_resolution=input_resolution,
                shift_size=0 if (d % 2 == 0) else window_size // 2,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                fused_window_process=fused_window_process,
                ##
                continuous_scan=continuous_scan,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))
        
        self.blocks =  nn.ModuleList(blocks)



    def forward(self, x, c):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, c)
            else:
                x = blk(x, c)

        return x


class LaMambaDiff(nn.Module):
    def __init__(
        self, 
        patch_size=2, 
        in_chans=4, 
        num_classes=1000, 
        depths=[2, 2, 4, 2],
        bottleneck_depth=1, 
        dims=[96, 192, 384, 384], 
        unconditional=True, # classifier free guidance
        sd_unet_decoder_design=False,
        skip_concat=False,
        # =========================
        in_resolution=32,
        num_heads=[8,8,8,8],
        window_size=8,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        fused_window_process=False, 
        # =========================
        class_dropout_prob=0.1,
        learn_sigma=True,
        # =========================
        continuous_scan=True,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        print('channel first:', self.channel_first)
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers-1)]
            dims = dims+[dims[-1]]
        self.num_features = dims[-1]
        self.dims = dims
        self.dims_decoder = [dims[-(i+1)] for i in range(len(dims))]
        

        dpr = [0 for _ in range(sum(depths))]
        # following unet in stable diffusion, decoder is larger by one block for each stage
        if sd_unet_decoder_design:
            if depths[-1] == 0:
                depth_up = [0]+[depths[-(i+1)]+1 for i in range(1,len(depths))]
            else:
                depth_up = [depths[-(i+1)]+1 for i in range(len(depths))]
        else:
            depth_up = [depths[-(i+1)] for i in range(len(depths))]

        dpr_up = [0 for _ in range(sum(depth_up))]
        num_heads_up = [num_heads[-(i+1)] for i in range(len(num_heads))]

        # learnable position embedding:
        self.num_patches = in_resolution*in_resolution
        self.pos_embed = nn.Parameter(torch.zeros(1, in_resolution, in_resolution, dims[0]), requires_grad=True)

        # diffusion timestep and condition embedder
        condition_dim = dims[-1]
        self.condition_dim = condition_dim
        self.t_embedder = TimestepEmbedder(condition_dim)
        
        # For unconditional diffusion model, mask all labels
        if unconditional:
            class_dropout_prob=1.0
            assert num_classes==0, 'Number of class not 0 for unconditional training'
        self.y_embedder = LabelEmbedder(num_classes, condition_dim, class_dropout_prob)
       

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.patch_embed = PatchEmbed2D_(patch_size=patch_size, in_chans=in_chans, embed_dim=dims[0],
            norm_layer=norm_layer if patch_norm else None)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        # encoder blocks (down)
        self.layers = nn.ModuleList()
        last_resolution=None
        skip = 2  
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - skip) else nn.Identity() # only use downsample for the first two stages e.g. (32 -> 16 -> 8 -> 8)

            bottleneck_stage = i_layer==self.num_layers -1
            self.layers.append(LaMambaDiff_layer(
                dim = self.dims[i_layer],
                condition_dim=condition_dim,
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                bottleneck_stage=bottleneck_stage,
                # =================
                input_resolution=in_resolution // 2**(i_layer) if (i_layer < self.num_layers - skip) else in_resolution // 4, 
                window_size=window_size,
                num_heads=num_heads[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                fused_window_process=fused_window_process,                # =================
                continuous_scan=continuous_scan,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))
            last_resolution= in_resolution// 2**(i_layer+1)


        # mid block
        self.mid_layer = LaMambaDiff_bottleneck(
                dim = self.dims[i_layer],
                condition_dim=condition_dim,
                drop_path = None,
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                channel_first=self.channel_first,
                # =================
                input_resolution=in_resolution // 4,
                window_size=window_size,
                num_heads=num_heads[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                fused_window_process=fused_window_process,
                bottleneck_depth=bottleneck_depth,
                # =================
                continuous_scan=continuous_scan,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            )

        # decoder blocks (up)
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            skip_up= 1 # final
            if i_layer <= skip_up:
                upsample=None
            else:
                upsample = PatchExpanding2D(dim = self.dims_decoder[i_layer-1],dim_scale=2,norm_layer=norm_layer, channel_first=self.channel_first)


            dprs = dpr_up[sum(depth_up[:i_layer]):sum(depth_up[:i_layer + 1])]

            bottleneck_stage = i_layer==0
            residual_dims = 0
            
            self.layers_up.append(LaMambaDiff_layer_up(
                dim = self.dims_decoder[i_layer],
                upsample = upsample,
                condition_dim=condition_dim,
                drop_path = dprs,
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                channel_first=self.channel_first,
                bottleneck_stage=bottleneck_stage,
                residual_dims=residual_dims,
                skip_concat=skip_concat,
                full_condition_project=True,
                # =================
                input_resolution= in_resolution // 4 if (i_layer < skip_up) else in_resolution // 2**(self.num_layers-i_layer-skip_up),
                window_size=window_size,
                num_heads=num_heads_up[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                fused_window_process=fused_window_process,
                # =================
                continuous_scan=continuous_scan,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                decoder=False,
            ))


        self.out_channels = in_chans
        self.final_up = FinalPatchExpanding2D(dim=self.dims_decoder[-1], dim_scale=patch_size, norm_layer=norm_layer)
        self.final_condition_projection = nn.Linear(self.condition_dim, self.dims_decoder[-1],bias=True)
        self.norm_final = norm_layer(self.dims_decoder[-1], elementwise_affine=False, eps=1e-6)
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dims_decoder[-1], 2 * self.dims_decoder[-1], bias=True)
        )

        self.final_linear = nn.Linear(self.dims_decoder[-1], self.out_channels*2, bias=True)


        self._init_weights()
    def _init_weights(self):
        # Initialize ssm layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Initialize pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks. Note some bottleneck blocks do not have conditions, skip them:
        for block in self.layers:
            for ssm in block.blocks:
                try:
                    nn.init.constant_(ssm.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(ssm.adaLN_modulation[-1].bias, 0)
                except:
                    pass
                
        for ssm in self.mid_layer.blocks:
            try:
                nn.init.constant_(ssm.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(ssm.adaLN_modulation[-1].bias, 0)
            except:
                pass

        for block in self.layers_up:
            for ssm in block.blocks:
                try:
                    nn.init.constant_(ssm.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(ssm.adaLN_modulation[-1].bias, 0)
                except:
                    pass

        # Zero-out output layers:
        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)

        for block in self.layers_up:
            try:
                nn.init.constant_(block.weight, 0)
                nn.init.constant_(block.bias, 0)
            except:
                pass
        



    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )


    def forward_features(self, x, c):
        skip_list = []
        x = self.patch_embed(x) + self.pos_embed # fixed positional embedding

        for layer in self.layers:
            # note the output of the last stage (bottleneck) is not stored in skip_list
            x, skip_feat = layer(x, c)
            skip_list.append(skip_feat)
        return x, skip_list
    
    def forward_features_up(self, x, c, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            x = layer_up(x, c, skip_list[-(inx+1)])
        return x
    
    def forward_final(self, x, c):

        c = self.final_condition_projection(c)[:,None,None,:]
        x = self.final_up(x)
        shift, scale = self.final_adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.final_linear(x)
 

        x = x.permute(0,3,1,2)
        return x

    def forward(self, x, t, y):
        """
        Forward pass.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        # encoder (down) blocks
        x, skip_list = self.forward_features(x, c) 

        # mid
        x = x + self.mid_layer(x, c)
        bottleneck = x  

        # decoder (up) blocks              
        x = self.forward_features_up(x, c, skip_list)

        x = self.forward_final(x, c)
        
        return x


    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_chans], model_out[:, self.in_chans:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


    def flops(self, resolution=256):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::gelu": None, # as relu is in _IGNORED_OPS
            "aten::index_select": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit}

        model = copy.deepcopy(self)
        model.cuda().eval()

        latent_size = int(resolution/8)
        latent_input= torch.randn([1,4,latent_size,latent_size],device=next(model.parameters()).device)
        timestamp_input= torch.randn([1],device=next(model.parameters()).device)
        condition_input= torch.zeros([1],device=next(model.parameters()).device, dtype=torch.long)

        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(latent_input,timestamp_input,condition_input,), supported_ops=supported_ops)

        del model, latent_input, timestamp_input, condition_input
        return sum(Gflops.values()) 

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
