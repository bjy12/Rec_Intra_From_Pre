"""
Pure U-Net Denoising Network for Latent Space Diffusion

纯 U-Net 版本的去噪网络，不包含 DiT 组件。
用于 latent space 扩散模型的术中CT重建。

输入:
- x: [B, 8, 32, 32, 32] 噪声化的 latent
- time: [B] 时间步
- condition_dict: 条件信息 (pre_latent, drr_images, level_idx)

输出:
- predicted_noise: [B, 8, 32, 32, 32]
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from einops import rearrange

from ddpm.model.latent_condition_model import LatentConditionModel


# ============================================================================
# Helper Functions
# ============================================================================

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# ============================================================================
# Basic Modules
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """正弦位置编码用于时间步嵌入"""
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


class LayerNorm3D(nn.Module):
    """3D LayerNorm"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm3D(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


# ============================================================================
# U-Net Blocks
# ============================================================================

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    """带时间条件的 ResNet Block"""
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """3D Self-Attention Block"""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x_flat = rearrange(x, 'b c d h w -> b (d h w) c').contiguous()
        qkv = self.to_qkv(x_flat).chunk(3, dim=2)
        q, k, v = [rearrange(t, 'b n (h c) -> b h n c', h=self.heads) for t in qkv]
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = rearrange(out, 'b h (d x y) c -> b (h c) d x y', d=d, x=h, y=w).contiguous()
        return self.to_out(out)


# ============================================================================
# Main Model
# ============================================================================

class UNet3D_Latent(nn.Module):
    """
    纯 U-Net 3D 去噪网络 (无 DiT 组件)
    
    专为 latent space 扩散设计，适用于 32³ 的 latent 体积。
    
    Args:
        dim: 基础通道数
        channels: 输入 latent 通道数 (默认 8)
        condition_channels: 条件特征通道数 (默认 128)
        dim_mults: 各层通道倍数
        use_attention: 各层是否使用注意力
        latent_channels: CT latent 通道数
        latent_size: latent 空间尺寸
    """
    def __init__(
        self,
        dim=64,
        channels=8,
        condition_channels=128,
        dim_mults=(1, 2, 4, 8),
        use_attention=(False, False, True, True),
        attn_heads=4,
        resnet_groups=8,
        latent_channels=8,
        latent_size=32,
    ):
        super().__init__()
        
        self.channels = channels
        self.dim = dim
        
        # 条件分支
        self.condition_branch = LatentConditionModel(
            latent_channels=latent_channels,
            drr_in_channels=1,
            base_channels=32,
            feat_channels=condition_channels,
            out_channels=condition_channels,
            latent_size=latent_size,
        )
        
        # 输入通道 = latent 通道 + 条件通道
        input_channels = channels + condition_channels
        
        # 初始卷积
        self.init_conv = nn.Conv3d(input_channels, dim, 3, padding=1)
        
        # 时间嵌入
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 构建 U-Net 层
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
        
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)
        
        # Encoder
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (num_resolutions - 1)
            use_attn = use_attention[ind] if ind < len(use_attention) else False
            
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, AttentionBlock(dim_out, heads=attn_heads))) if use_attn else nn.Identity(),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
        
        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, AttentionBlock(mid_dim, heads=attn_heads)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
        
        # Decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (num_resolutions - 1)
            use_attn = use_attention[num_resolutions - 1 - ind] if (num_resolutions - 1 - ind) < len(use_attention) else False
            
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_out),  # skip connection
                block_klass_cond(dim_out, dim_in),
                Residual(PreNorm(dim_in, AttentionBlock(dim_in, heads=attn_heads))) if use_attn else nn.Identity(),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        
        # 输出
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),  # 包含初始特征的 skip
            nn.Conv3d(dim, channels, 1)
        )
    
    def forward(self, x, time, condition_dict=None):
        """
        Args:
            x: [B, 8, 32, 32, 32] 噪声化的 latent
            time: [B] 时间步
            condition_dict: 条件字典
        
        Returns:
            predicted_noise: [B, 8, 32, 32, 32]
        """
        # 获取条件特征
        if condition_dict is not None:
            hint = self.condition_branch(condition_dict)  # [B, 128, 32, 32, 32]
        else:
            raise ValueError("condition_dict cannot be None")
        
        # 拼接输入
        x = torch.cat((x, hint), dim=1)  # [B, 8+128, 32, 32, 32]
        
        # 初始卷积
        x = self.init_conv(x)
        r = x.clone()  # 保存用于最终 skip
        
        # 时间嵌入
        t = self.time_mlp(time)
        
        # Encoder
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        
        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # Decoder
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        
        # 最终输出
        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing UNet3D_Latent...")
    
    model = UNet3D_Latent(
        dim=64,
        channels=8,
        condition_channels=128,
        dim_mults=(1, 2, 4),
        use_attention=(False, True, True),
        latent_channels=8,
        latent_size=32,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试
    B = 2
    x = torch.randn(B, 8, 32, 32, 32)
    time = torch.randint(0, 1000, (B,))
    condition_dict = {
        'pre_latent': torch.randn(B, 8, 32, 32, 32),
        'drr_images': torch.randn(B, 2, 1, 128, 128),
        'level_idx': torch.tensor([0, 3]),
    }
    
    with torch.no_grad():
        out = model(x, time, condition_dict)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        time = time.cuda()
        condition_dict = {k: v.cuda() for k, v in condition_dict.items()}
        
        with torch.no_grad():
            out = model(x, time, condition_dict)
        
        print(f"\nGPU test passed!")
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
