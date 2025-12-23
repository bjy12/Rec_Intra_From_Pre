"""
Multi-Task Condition Branch Model V3 (Simplified Baseline)

简化版本：
1. DRR Encoder: 2D ResNet → [B, 2, C, 32, 32]
2. CT Encoder: 3D ResBlocks → [B, C, 32, 32, 32]
3. 几何引导的 2D→3D Lift: 利用已知投影角度
4. 简单 Concat + Conv 融合
5. Dual-Branch: Generation + Registration

设计原则：
- 简单但有效
- 易于调试和迭代
- 显存友好
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


# ============================================================================
# 基础模块
# ============================================================================

class ResBlock3D(nn.Module):
    """3D Residual Block with optional downsampling"""
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
    """2D Residual Block with optional downsampling"""
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


class CTEncoder3D(nn.Module):
    """
    3D ResBlocks for CT feature extraction
    Input: [B, 1, 128, 128, 128] → Output: [B, C, 32, 32, 32]
    """
    def __init__(self, in_ch=1, base_ch=32, out_ch=128):
        super().__init__()
        self.out_ch = out_ch
        
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(base_ch),
            nn.GELU(),
        )
        
        self.layer1 = ResBlock3D(base_ch, base_ch * 2, stride=2)      # 128→64
        self.layer2 = ResBlock3D(base_ch * 2, base_ch * 4, stride=2)  # 64→32
        self.layer3 = ResBlock3D(base_ch * 4, out_ch, stride=1)       # 32→32
    
    def forward(self, ct: torch.Tensor) -> torch.Tensor:
        x = self.stem(ct)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ============================================================================
# Geometry-Guided 2D→3D Lift
# ============================================================================

class GeometryGuidedLift(nn.Module):
    """
    几何引导的 2D→3D 特征提升
    
    原理：
    - AP view (正位, angle≈0): 特征沿 Y 轴 (前后方向) broadcast
    - Lateral view (侧位, angle≈π/2): 特征沿 X 轴 (左右方向) broadcast
    
    然后学习如何融合两个视角的信息。
    """
    def __init__(self, in_ch: int, out_ch: int, volume_size: int = 32):
        super().__init__()
        self.volume_size = volume_size
        
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
            volume_feat: [B, C, D, H, W] - 3D 特征体积
        """
        B, V, C, H, W = drr_feat.shape
        D = self.volume_size
        
        # View 0 (AP view): 特征沿 depth 维度扩展
        feat_ap = drr_feat[:, 0]  # [B, C, H, W]
        feat_ap = self.view_refine(feat_ap)
        
        # 学习深度权重
        depth_w_ap = self.depth_weights(feat_ap)  # [B, D, H, W]
        
        # 软 broadcast: 特征 × 深度权重
        # feat_ap: [B, C, H, W] → [B, C, 1, H, W]
        # depth_w_ap: [B, D, H, W] → [B, 1, D, H, W]
        feat_ap_3d = feat_ap.unsqueeze(2) * depth_w_ap.unsqueeze(1)  # [B, C, D, H, W]
        
        # View 1 (Lateral view): 类似处理
        feat_lat = drr_feat[:, 1]
        feat_lat = self.view_refine(feat_lat)
        depth_w_lat = self.depth_weights(feat_lat)
        
        # 侧位的"深度"对应 X 方向，需要 permute
        # [B, C, H, W] → 视为 [B, C, D, H] where D=W
        # 最终要得到 [B, C, D, H, W]
        feat_lat_3d = feat_lat.unsqueeze(-1) * depth_w_lat.unsqueeze(1).permute(0, 1, 3, 2, 4)
        # 这里简化处理：直接用同样的方式
        feat_lat_3d = feat_lat.unsqueeze(2) * depth_w_lat.unsqueeze(1)  # [B, C, D, H, W]
        feat_lat_3d = feat_lat_3d.permute(0, 1, 4, 3, 2)  # 调整轴顺序
        
        # 确保大小一致
        if feat_lat_3d.shape != feat_ap_3d.shape:
            feat_lat_3d = F.interpolate(feat_lat_3d, size=feat_ap_3d.shape[2:], mode='trilinear', align_corners=True)
        
        # 融合两个视角
        fused = torch.cat([feat_ap_3d, feat_lat_3d], dim=1)  # [B, 2C, D, H, W]
        fused = self.fusion(fused)  # [B, C, D, H, W]
        
        return fused


# ============================================================================
# Simple Fusion Module
# ============================================================================

class SimpleFusion(nn.Module):
    """
    简单的 Concat + Conv 融合
    比 Cross-Attention 更简单稳定
    """
    def __init__(self, drr_ch: int, ct_ch: int, out_ch: int):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Conv3d(drr_ch + ct_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            ResBlock3D(out_ch, out_ch),
            ResBlock3D(out_ch, out_ch),
        )
    
    def forward(self, drr_3d: torch.Tensor, ct_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drr_3d: [B, C_drr, D, H, W]
            ct_feat: [B, C_ct, D, H, W]
        Returns:
            fused: [B, C_out, D, H, W]
        """
        x = torch.cat([drr_3d, ct_feat], dim=1)
        return self.fusion(x)


# ============================================================================
# Decoder and Heads
# ============================================================================

class GenerationDecoder(nn.Module):
    """上采样解码器: 32³ → 128³"""
    def __init__(self, in_ch=128, base_ch=64):
        super().__init__()
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(in_ch, base_ch * 2, 4, stride=2, padding=1),  # 32→64
            nn.BatchNorm3d(base_ch * 2),
            nn.GELU(),
            ResBlock3D(base_ch * 2, base_ch * 2),
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(base_ch * 2, base_ch, 4, stride=2, padding=1),  # 64→128
            nn.BatchNorm3d(base_ch),
            nn.GELU(),
            ResBlock3D(base_ch, base_ch),
        )
        
        self.out = nn.Sequential(
            nn.Conv3d(base_ch, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        return self.out(x)


class RegistrationHead(nn.Module):
    """预测 6-DoF 变换参数"""
    def __init__(self, in_ch=128, hidden_dim=256):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)


# ============================================================================
# Main Model
# ============================================================================

class MultiTaskConditionBranchV3(nn.Module):
    """
    简化版多任务条件分支网络
    
    特点：
    1. 简单的 2D/3D 编码器
    2. 几何引导的 2D→3D lift (学习深度分布)
    3. 简单的 concat + conv 融合
    4. 双分支输出
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        feat_channels: int = 128,
        num_attention_heads: int = 4,  # 保留参数兼容性，但不使用
        max_rotation_deg: float = 20.0,
        max_translation_voxels: float = 10.0,
    ):
        super().__init__()
        
        self.max_rotation_rad = np.deg2rad(max_rotation_deg)
        self.max_translation = max_translation_voxels
        
        # Encoders
        self.drr_encoder = DRREncoder2D(
            in_ch=in_channels,
            base_ch=base_channels,
            out_ch=feat_channels,
        )
        
        self.ct_encoder = CTEncoder3D(
            in_ch=in_channels,
            base_ch=base_channels,
            out_ch=feat_channels,
        )
        
        # 2D → 3D Lift with geometry guidance
        self.lift = GeometryGuidedLift(
            in_ch=feat_channels,
            out_ch=feat_channels,
            volume_size=32,
        )
        
        # Fusion
        self.fusion = SimpleFusion(
            drr_ch=feat_channels,
            ct_ch=feat_channels,
            out_ch=feat_channels,
        )
        
        # Heads
        self.generation_decoder = GenerationDecoder(
            in_ch=feat_channels,
            base_ch=base_channels * 2,
        )
        
        self.registration_head = RegistrationHead(
            in_ch=feat_channels,
        )
    
    def forward(
        self,
        perturbed_ct: torch.Tensor,
        drr_images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            perturbed_ct: [B, 1, 128, 128, 128]
            drr_images: [B, 2, 1, 128, 128]
        Returns:
            dict with transform_pred, intra_ct_pred, condition_feature
        """
        # Encode
        drr_feat = self.drr_encoder(drr_images)    # [B, 2, 128, 32, 32]
        ct_feat = self.ct_encoder(perturbed_ct)    # [B, 128, 32, 32, 32]
        
        # Lift DRR 2D → 3D with geometry guidance
        drr_3d = self.lift(drr_feat)  # [B, 128, 32, 32, 32]
        
        # Fusion
        fused = self.fusion(drr_3d, ct_feat)  # [B, 128, 32, 32, 32]
        
        # Heads
        intra_ct_pred = self.generation_decoder(fused)   # [B, 1, 128, 128, 128]
        transform_pred = self.registration_head(fused)   # [B, 6]
        
        return {
            'transform_pred': transform_pred,
            'intra_ct_pred': intra_ct_pred,
            'condition_feature': fused,
        }


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    model = MultiTaskConditionBranchV3(
        in_channels=1,
        base_channels=32,
        feat_channels=128,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    B = 1
    perturbed_ct = torch.randn(B, 1, 128, 128, 128)
    drr_images = torch.randn(B, 2, 1, 128, 128)
    
    if torch.cuda.is_available():
        model = model.cuda()
        perturbed_ct = perturbed_ct.cuda()
        drr_images = drr_images.cuda()
    
    with torch.no_grad():
        output = model(perturbed_ct, drr_images)
    
    print("transform_pred shape:", output['transform_pred'].shape)
    print("intra_ct_pred shape:", output['intra_ct_pred'].shape)
    print("condition_feature shape:", output['condition_feature'].shape)
    
    # 显存估计
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
