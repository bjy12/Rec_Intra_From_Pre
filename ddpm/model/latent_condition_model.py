"""
Latent Space Condition Model for Pre-Intra CT Reconstruction

用于处理 latent space 数据的条件模型：
- 输入: 术前 CT latent + 术中 DRR 图像
- 输出: 融合的条件特征，用于扩散模型

设计特点：
1. CT Latent Encoder: 轻量级 3D 卷积，保持 latent 分辨率
2. DRR Encoder: 2D ResNet 提取特征
3. Geometry-Guided Lift: 2D→3D 特征提升
4. Simple Fusion: concat + conv 融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


# ============================================================================
# 基础模块
# ============================================================================

class ResBlock3D(nn.Module):
    """3D Residual Block with optional channel change"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()
        
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act(out)


class ResBlock2D(nn.Module):
    """2D Residual Block with optional channel change"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act(out)


# ============================================================================
# Encoders
# ============================================================================

class LatentCTEncoder(nn.Module):
    """
    轻量级 3D 编码器，用于处理 CT latent 特征
    
    Input: [B, C_latent, D, H, W] (如 [B, 8, 32, 32, 32])
    Output: [B, out_ch, D, H, W] (如 [B, 128, 32, 32, 32])
    
    注意：保持空间分辨率不变，只做通道变换和特征提取
    """
    def __init__(self, in_ch=8, mid_ch=64, out_ch=128):
        super().__init__()
        
        self.net = nn.Sequential(
            # 初始特征提取
            nn.Conv3d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.GELU(),
            # ResBlocks 进行特征提取
            ResBlock3D(mid_ch, mid_ch),
            ResBlock3D(mid_ch, out_ch),
            ResBlock3D(out_ch, out_ch),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W] CT latent
        Returns:
            [B, out_ch, D, H, W]
        """
        return self.net(x)


class DRREncoder2D(nn.Module):
    """
    2D ResNet for DRR feature extraction
    
    Input: [B, 2, 1, 128, 128] → Output: [B, 2, C, 32, 32]
    """
    def __init__(self, in_ch=1, base_ch=32, out_ch=128):
        super().__init__()
        self.out_ch = out_ch
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, stride=2, padding=3, bias=False),  # 128→64
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
        )
        
        self.layer1 = ResBlock2D(base_ch, base_ch * 2, stride=2)      # 64→32
        self.layer2 = ResBlock2D(base_ch * 2, base_ch * 4, stride=1)  # 32→32
        self.layer3 = ResBlock2D(base_ch * 4, out_ch, stride=1)       # 32→32
    
    def forward(self, drr_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drr_images: [B, 2, 1, 128, 128]
        Returns:
            features: [B, 2, C, 32, 32]
        """
        B, V, C, H, W = drr_images.shape
        x = drr_images.view(B * V, C, H, W)
        
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        _, C_out, H_out, W_out = x.shape
        return x.view(B, V, C_out, H_out, W_out)


# ============================================================================
# 2D → 3D Lift
# ============================================================================

class GeometryGuidedLift(nn.Module):
    """
    几何引导的 2D→3D 特征提升
    
    原理：
    - AP view (正位): 特征沿深度维度 broadcast
    - Lateral view (侧位): 特征沿宽度维度 broadcast
    
    然后学习如何融合两个视角的信息。
    """
    def __init__(self, in_ch: int, out_ch: int, volume_size: int = 32):
        super().__init__()
        self.volume_size = volume_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # 每个视角的特征先经过 2D conv 提取
        self.view_refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
        )
        
        # 学习深度维度的权重分布 (软 broadcast)
        self.depth_weights = nn.Sequential(
            nn.Conv2d(in_ch, volume_size, 1),  # 预测每个深度的权重
            nn.Softmax(dim=1),  # 归一化
        )
        
        # 融合两个视角
        self.fusion = nn.Sequential(
            nn.Conv3d(in_ch * 2, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )
    
    def forward(self, drr_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drr_feat: [B, 2, C, H, W] - 两个视角的 DRR 特征
        Returns:
            volume_feat: [B, out_ch, D, H, W] - 3D 特征体积
        """
        B, V, C, H, W = drr_feat.shape
        D = self.volume_size
        
        # View 0 (AP view): 特征沿 depth 维度扩展
        feat_ap = drr_feat[:, 0]  # [B, C, H, W]
        feat_ap = self.view_refine(feat_ap)
        
        # 学习深度权重
        depth_w_ap = self.depth_weights(feat_ap)  # [B, D, H, W]
        
        # 软 broadcast: 特征 × 深度权重
        feat_ap_3d = feat_ap.unsqueeze(2) * depth_w_ap.unsqueeze(1)  # [B, C, D, H, W]
        
        # View 1 (Lateral view): 类似处理
        feat_lat = drr_feat[:, 1]
        feat_lat = self.view_refine(feat_lat)
        depth_w_lat = self.depth_weights(feat_lat)
        
        # 侧位视角，同样的方式处理
        feat_lat_3d = feat_lat.unsqueeze(2) * depth_w_lat.unsqueeze(1)  # [B, C, D, H, W]
        # 调整轴顺序以对应不同的投影方向
        feat_lat_3d = feat_lat_3d.permute(0, 1, 4, 3, 2)  # [B, C, W, H, D] → 视为 [B, C, D', H', W']
        
        # 确保大小一致
        if feat_lat_3d.shape != feat_ap_3d.shape:
            feat_lat_3d = F.interpolate(
                feat_lat_3d, 
                size=feat_ap_3d.shape[2:], 
                mode='trilinear', 
                align_corners=True
            )
        
        # 融合两个视角
        fused = torch.cat([feat_ap_3d, feat_lat_3d], dim=1)  # [B, 2C, D, H, W]
        fused = self.fusion(fused)  # [B, out_ch, D, H, W]
        
        return fused


# ============================================================================
# Fusion Module
# ============================================================================

class LatentFusion(nn.Module):
    """
    融合 CT latent 特征和 DRR 3D 特征
    使用简单的 Concat + Conv 策略
    """
    def __init__(self, ct_ch: int, drr_ch: int, out_ch: int):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Conv3d(ct_ch + drr_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            ResBlock3D(out_ch, out_ch),
            ResBlock3D(out_ch, out_ch),
        )
    
    def forward(self, ct_feat: torch.Tensor, drr_3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ct_feat: [B, C_ct, D, H, W] - CT latent 特征
            drr_3d: [B, C_drr, D, H, W] - DRR 3D 特征
        Returns:
            fused: [B, out_ch, D, H, W]
        """
        # 确保尺寸匹配
        if ct_feat.shape[2:] != drr_3d.shape[2:]:
            drr_3d = F.interpolate(
                drr_3d, 
                size=ct_feat.shape[2:], 
                mode='trilinear', 
                align_corners=True
            )
        
        x = torch.cat([ct_feat, drr_3d], dim=1)
        return self.fusion(x)


# ============================================================================
# Main Model
# ============================================================================

class LatentConditionModel(nn.Module):
    """
    用于 Latent Space 扩散模型的条件分支
    
    输入:
        - pre_latent: [B, C, D, H, W] 术前 CT latent (如 [B, 8, 32, 32, 32])
        - drr_images: [B, 2, 1, 128, 128] 术中 DRR 图像
        - level: [B] 脊柱级别 (0-4 对应 L1-L5)
    
    输出:
        - condition_feature: [B, out_ch, D, H, W] 融合的条件特征
    
    用于传递给扩散模型的去噪网络。
    """
    def __init__(
        self,
        latent_channels: int = 8,        # CT latent 的通道数
        drr_in_channels: int = 1,        # DRR 输入通道
        base_channels: int = 32,         # 基础通道数
        feat_channels: int = 128,        # 中间特征通道数
        out_channels: int = 128,         # 输出条件特征通道数
        latent_size: int = 32,           # latent 空间尺寸
        num_levels: int = 5,             # 脊柱级别数量 (L1-L5)
        level_embed_dim: int = 64,       # level embedding 维度
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.num_levels = num_levels
        self.level_embed_dim = level_embed_dim
        
        # CT Latent Encoder
        self.ct_encoder = LatentCTEncoder(
            in_ch=latent_channels,
            mid_ch=base_channels * 2,
            out_ch=feat_channels,
        )
        
        # DRR Encoder
        self.drr_encoder = DRREncoder2D(
            in_ch=drr_in_channels,
            base_ch=base_channels,
            out_ch=feat_channels,
        )
        
        # 2D → 3D Lift
        self.lift = GeometryGuidedLift(
            in_ch=feat_channels,
            out_ch=feat_channels,
            volume_size=latent_size,
        )
        
        # Fusion (包含 level embedding)
        # 融合时额外加入 level embedding 通道
        self.fusion = LatentFusion(
            ct_ch=feat_channels,
            drr_ch=feat_channels,
            out_ch=out_channels,
        )
        
        # Level Embedding: 脊柱级别嵌入
        self.level_embedding = nn.Embedding(num_levels, level_embed_dim)
        
        # 将 level embedding 投影并与融合特征结合
        self.level_proj = nn.Sequential(
            nn.Linear(level_embed_dim, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        
        self.all_vertebral_level = ["L1", "L2", "L3", "L4", "L5"]

        # 最终融合层（结合 level 信息）
        self.final_fusion = nn.Sequential(
            ResBlock3D(out_channels, out_channels),
            ResBlock3D(out_channels, out_channels),
        )
    
    def forward(self, condition_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            condition_dict: {
                'pre_latent': [B, C, D, H, W],  # 术前 CT latent
                'drr_images': [B, 2, 1, H, W],  # DRR 图像
                'level': [B],                   # 脊柱级别 (0-4)
            }
        
        Returns:
            condition_feature: [B, out_ch, D, H, W]
        """
        pre_latent = condition_dict['pre_latent']  # [B, C, D, H, W]
        drr_images = condition_dict['drr_images']  # [B, 2, 1, 128, 128]
        level = condition_dict.get('level_idx', None)  # [B] 脊柱级别索引 (0-4)

        # 编码 CT latent
        ct_feat = self.ct_encoder(pre_latent)  # [B, feat_ch, D, H, W]
        
        # 编码 DRR
        drr_feat = self.drr_encoder(drr_images)  # [B, 2, feat_ch, 32, 32]
        
        # 2D → 3D 提升
        drr_3d = self.lift(drr_feat)  # [B, feat_ch, 32, 32, 32]
        
        # 融合 CT 和 DRR 特征
        condition_feature = self.fusion(ct_feat, drr_3d)  # [B, out_ch, D, H, W]
        
        # 添加 Level 条件
        if level is not None:
            B = condition_feature.shape[0]
            device = condition_feature.device
            
            
            level_emb = self.level_embedding(level)  # [B, level_embed_dim]
            level_proj = self.level_proj(level_emb)  # [B, out_channels]
            
            # 空间广播: [B, C] -> [B, C, D, H, W]
            level_spatial = level_proj.view(B, -1, 1, 1, 1).expand_as(condition_feature)
            
            # 特征调制 (加法融合)
            condition_feature = condition_feature + level_spatial
        
        # 最终融合
        condition_feature = self.final_fusion(condition_feature)
        
        return condition_feature


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing LatentConditionModel...")
    
    # 创建模型
    model = LatentConditionModel(
        latent_channels=8,
        drr_in_channels=1,
        base_channels=32,
        feat_channels=128,
        out_channels=128,
        latent_size=32,
        num_levels=5,  # L1-L5
        level_embed_dim=64,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试输入 (包含 level)
    B = 2
    condition_dict = {
        'pre_latent': torch.randn(B, 8, 32, 32, 32),
        'drr_images': torch.randn(B, 2, 1, 128, 128),
        'level': ["L1", "L4"],  # L1 和 L4
    }
    
    # 前向传播
    with torch.no_grad():
        output = model(condition_dict)
    
    print(f"Input pre_latent shape: {condition_dict['pre_latent'].shape}")
    print(f"Input drr_images shape: {condition_dict['drr_images'].shape}")
    print(f"Input level: {condition_dict['level']} (L1, L4)")
    print(f"Output condition_feature shape: {output.shape}")
    
    # 测试不带 level 的情况
    condition_dict_no_level = {
        'pre_latent': torch.randn(B, 8, 32, 32, 32),
        'drr_images': torch.randn(B, 2, 1, 128, 128),
    }
    with torch.no_grad():
        output_no_level = model(condition_dict_no_level)
    print(f"Output without level shape: {output_no_level.shape}")
    
    # GPU 测试
    if torch.cuda.is_available():
        model = model.cuda()
        condition_dict = {k: v.cuda() for k, v in condition_dict.items()}
        
        with torch.no_grad():
            output = model(condition_dict)
        
        print(f"\nGPU test passed!")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

