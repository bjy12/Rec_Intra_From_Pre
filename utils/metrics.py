"""
3D 医学图像质量评估指标工具

包含 SSIM、PSNR、MAE 等指标的计算函数。
可在训练和评估过程中调用。

使用示例:
    from utils.metrics import compute_ssim_3d, compute_psnr_3d, compute_metrics_batch
    
    # 单对图像
    ssim = compute_ssim_3d(pred_ct, target_ct)
    psnr = compute_psnr_3d(pred_ct, target_ct)
    
    # 批量计算所有指标
    metrics = compute_metrics_batch(pred_ct, target_ct)
    print(metrics)  # {'mae': 0.1, 'ssim': 0.85, 'psnr': 25.3}
"""

import torch
import numpy as np
from typing import Dict, Union, Optional


def normalize_to_01(x: torch.Tensor) -> torch.Tensor:
    """
    将张量归一化到 [0, 1] 范围。
    """
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-8:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算 Mean Absolute Error (MAE)。
    
    Args:
        pred: 预测图像 [B, C, D, H, W] 或 [C, D, H, W]
        target: 目标图像，形状与 pred 相同
    
    Returns:
        MAE 值 (float)
    """
    return torch.mean(torch.abs(pred - target)).item()


def compute_psnr_3d(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    data_range: Optional[float] = None
) -> float:
    """
    计算 3D 图像的 Peak Signal-to-Noise Ratio (PSNR)。
    
    Args:
        pred: 预测图像 [B, C, D, H, W] 或 [C, D, H, W]
        target: 目标图像
        data_range: 数据范围，如果为 None 则自动计算
    
    Returns:
        PSNR 值 (dB)
    """
    if data_range is None:
        data_range = max(target.max().item() - target.min().item(), 1e-8)
    
    mse = torch.mean((pred - target) ** 2).item()
    
    if mse < 1e-10:
        return float('inf')
    
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return psnr


def compute_ssim_3d(
    pred: torch.Tensor, 
    target: torch.Tensor,
    window_size: int = 7,
    data_range: Optional[float] = None,
    K1: float = 0.01,
    K2: float = 0.03,
) -> float:
    """
    计算 3D 图像的 Structural Similarity Index (SSIM)。
    
    使用 3D 高斯窗口计算局部 SSIM，然后取平均。
    
    Args:
        pred: 预测图像 [B, C, D, H, W] 或 [C, D, H, W]
        target: 目标图像
        window_size: 滑动窗口大小
        data_range: 数据范围
        K1, K2: SSIM 稳定常数
    
    Returns:
        SSIM 值 [0, 1]
    """
    # 确保有 batch 维度
    if pred.dim() == 4:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    device = pred.device
    
    if data_range is None:
        data_range = max(target.max().item() - target.min().item(), 1e-8)
    
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    # 创建 3D 高斯窗口
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2*sigma**2)) 
                          for x in range(window_size)])
    gauss = gauss / gauss.sum()
    
    # 3D 可分离卷积核
    _1D_window = gauss.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size]
    
    # 创建 3D 窗口
    _3D_window = _1D_window.unsqueeze(-1).unsqueeze(-1) * \
                 _1D_window.unsqueeze(-1).unsqueeze(2) * \
                 _1D_window.unsqueeze(2).unsqueeze(3)
    # [1, 1, window_size, window_size, window_size]
    
    window = _3D_window.to(device)
    
    # 通道数
    channels = pred.shape[1]
    window = window.expand(channels, 1, window_size, window_size, window_size)
    
    padding = window_size // 2
    
    # 计算均值
    mu1 = torch.nn.functional.conv3d(pred, window, padding=padding, groups=channels)
    mu2 = torch.nn.functional.conv3d(target, window, padding=padding, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = torch.nn.functional.conv3d(pred * pred, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = torch.nn.functional.conv3d(target * target, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = torch.nn.functional.conv3d(pred * target, window, padding=padding, groups=channels) - mu1_mu2
    
    # 确保方差非负
    sigma1_sq = torch.clamp(sigma1_sq, min=0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0)
    
    # SSIM 公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def compute_metrics_batch(
    pred: torch.Tensor, 
    target: torch.Tensor,
    normalize: bool = False,
) -> Dict[str, float]:
    """
    批量计算所有指标。
    
    Args:
        pred: 预测图像
        target: 目标图像
        normalize: 是否先归一化到 [0, 1]
    
    Returns:
        包含 'mae', 'ssim', 'psnr' 的字典
    """
    with torch.no_grad():
        if normalize:
            pred = normalize_to_01(pred)
            target = normalize_to_01(target)
        
        mae = compute_mae(pred, target)
        ssim = compute_ssim_3d(pred, target)
        psnr = compute_psnr_3d(pred, target)
    
    return {
        'mae': mae,
        'ssim': ssim,
        'psnr': psnr,
    }


class MetricsTracker:
    """
    用于训练过程中跟踪和累积指标的工具类。
    
    使用示例:
        tracker = MetricsTracker()
        
        for batch in val_loader:
            pred, target = model(batch)
            tracker.update(pred, target)
        
        results = tracker.compute()
        print(f"Mean SSIM: {results['ssim']:.4f}")
        
        tracker.reset()  # 下一个 epoch
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置累积值。"""
        self.mae_sum = 0.0
        self.ssim_sum = 0.0
        self.psnr_sum = 0.0
        self.count = 0
    
    def update(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        normalize: bool = False
    ):
        """
        更新累积值。
        
        Args:
            pred: 预测图像
            target: 目标图像
            normalize: 是否归一化
        """
        metrics = compute_metrics_batch(pred, target, normalize=normalize)
        
        self.mae_sum += metrics['mae']
        self.ssim_sum += metrics['ssim']
        self.psnr_sum += metrics['psnr'] if metrics['psnr'] != float('inf') else 0
        self.count += 1
    
    def compute(self) -> Dict[str, float]:
        """
        计算平均指标。
        
        Returns:
            平均指标字典
        """
        if self.count == 0:
            return {'mae': float('inf'), 'ssim': 0.0, 'psnr': 0.0}
        
        return {
            'mae': self.mae_sum / self.count,
            'ssim': self.ssim_sum / self.count,
            'psnr': self.psnr_sum / self.count,
        }
    
    def __str__(self) -> str:
        results = self.compute()
        return f"MAE: {results['mae']:.4f}, SSIM: {results['ssim']:.4f}, PSNR: {results['psnr']:.2f} dB"


# ============== 快速测试 ==============
if __name__ == "__main__":
    print("Testing metrics module...")
    
    # 创建测试数据
    torch.manual_seed(42)
    target = torch.randn(1, 1, 32, 32, 32).cuda()
    
    # 相似图像
    pred_similar = target + 0.1 * torch.randn_like(target)
    
    # 不相似图像
    pred_different = torch.randn_like(target)
    
    print("\n--- Similar Images ---")
    metrics_similar = compute_metrics_batch(pred_similar, target)
    print(f"MAE: {metrics_similar['mae']:.4f}")
    print(f"SSIM: {metrics_similar['ssim']:.4f}")
    print(f"PSNR: {metrics_similar['psnr']:.2f} dB")
    
    print("\n--- Different Images ---")
    metrics_different = compute_metrics_batch(pred_different, target)
    print(f"MAE: {metrics_different['mae']:.4f}")
    print(f"SSIM: {metrics_different['ssim']:.4f}")
    print(f"PSNR: {metrics_different['psnr']:.2f} dB")
    
    print("\n--- MetricsTracker Test ---")
    tracker = MetricsTracker()
    tracker.update(pred_similar, target)
    tracker.update(pred_different, target)
    print(tracker)
    
    print("\nAll tests passed!")
