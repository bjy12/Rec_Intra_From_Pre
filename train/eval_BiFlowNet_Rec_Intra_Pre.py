"""
BiFlowNet_Rec_Intra_Pre 评估脚本

用于加载训练好的 checkpoint，在验证集上运行推理并计算 MAE 等指标。
适配 Pre_Intra_Latent_Dataset 和新的 condition_dict 格式。
"""

import os
import sys
import copy
import json
import argparse
import logging
from types import SimpleNamespace

import numpy as np
import torch
import torchio as tio
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
from tqdm import tqdm

# 项目内模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from ddpm.BiFlowNet_Rec_Intra_Pre import GaussianDiffusion, BiFlowNet_Rec_Intra_Pre
from AutoEncoder.model.PatchVolume import patchvolumeAE
from dataset.Pre_Intra_Latent_Dataset import Pre_Intra_Latent_Dataset
from utils.metrics import compute_metrics_batch, MetricsTracker

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def dict_to_sns(d):
    """递归将字典转换为 SimpleNamespace。"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_sns(i) for i in d]
    return d


def create_logger(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/eval.log")],
    )
    return logging.getLogger(__name__)


def load_cfg(args):
    """加载 YAML 配置并应用命令行覆盖。"""
    cfg_raw = OmegaConf.load(args.config)
    cfg = dict_to_sns(OmegaConf.to_container(cfg_raw, resolve=True))

    # 命令行覆盖
    if args.ckpt:
        cfg.model.ckpt = args.ckpt
    if args.AE_ckpt:
        cfg.model.AE_ckpt = args.AE_ckpt
    if args.val_files_names_path:
        cfg.model.val_files_names_path = args.val_files_names_path
    if args.results_dir:
        cfg.model.results_dir = args.results_dir
    if args.val_batch_size is not None:
        cfg.model.val_batch_size = args.val_batch_size

    return cfg


def build_model(cfg, device):
    """构建 BiFlowNet_Rec_Intra_Pre 模型。"""
    use_dit = getattr(cfg.model, 'use_dit', True)
    
    model = BiFlowNet_Rec_Intra_Pre(
        dim=cfg.model.model_dim,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.volume_channels,
        init_kernel_size=3,
        cond_classes=None,
        res_condition=False,
        learn_sigma=False,
        use_sparse_linear_attn=cfg.model.use_attn,
        vq_size=cfg.model.vq_size,
        num_mid_DiT=cfg.model.num_dit,
        patch_size=cfg.model.patch_size,
        latent_channels=cfg.model.volume_channels,
        latent_size=cfg.model.resolution[0],
        condition_channels=cfg.model.condition_channels,
        use_dit=use_dit,
    ).to(device)

    diffusion = GaussianDiffusion(
        channels=cfg.model.volume_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    ).to(device)

    if not cfg.model.AE_ckpt:
        raise ValueError("请提供 --AE-ckpt 或在配置中设置 model.AE_ckpt。")
    AE = patchvolumeAE.load_from_checkpoint(cfg.model.AE_ckpt).to(device)
    AE.eval()
    
    return model, diffusion, AE


def load_checkpoint(model, ema_model, ckpt_path, device, use_ema=True, logger=None):
    """加载训练好的 checkpoint。"""
    if ckpt_path is None:
        raise ValueError("必须提供 --ckpt 路径以进行验证。")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    if "ema" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema"], strict=True)
    else:
        ema_model.load_state_dict(model.state_dict(), strict=True)

    eval_model = ema_model if use_ema and "ema" in checkpoint else model
    if logger:
        logger.info(
            f"Loaded checkpoint from {ckpt_path} | use_ema={use_ema} "
            f"(epoch={checkpoint.get('epoch', 'NA')}, step={checkpoint.get('train_steps', 'NA')})"
        )
    return eval_model


def evaluate(model, diffusion, AE, val_loader, device, amp, save_dir=None, 
             save_limit=None, logger=None):
    """
    在验证集上运行评估。
    
    Returns:
        包含 MAE, SSIM, PSNR 等指标的字典
    """
    model.eval()
    latent_mae_list = []
    ct_mae_list = []
    
    # 新增：用于计算 SSIM 和 PSNR
    ct_metrics_tracker = MetricsTracker()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    


    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating")
        for idx, batch in pbar:
            # 解包数据 - 适配 Pre_Intra_Latent_Dataset
            pre_latent = batch['pre_latent'].to(device)
            intra_latent = batch['intra_latent'].to(device)  # GT
            drr_images = batch['drr_images'].to(device)
            level_idx = batch['level_idx'].to(device)
            names = batch.get('name', [f'sample_{idx}'])
            print( " eval_case_name: ", names , " level:" , level_idx)
            # 构建 condition_dict - 适配新格式
            condition_dict = {
                'pre_latent': pre_latent,
                'drr_images': drr_images,
                'level_idx': level_idx,
            }
            
            # 从纯噪声采样
            z_sample = torch.randn_like(intra_latent, device=device)
            
            with autocast(enabled=amp):
                gen_latent = diffusion.p_sample_loop(model, z_sample, condition_dict=condition_dict)

            # ======== Latent 空间 MAE ========
            latent_mae = torch.mean(torch.abs(gen_latent - intra_latent)).item()
            latent_mae_list.append(latent_mae)

            # ======== 解码到 CT 空间 ========
            # 还原到 VQ codebook 范围
            gen_latent_scaled = (((gen_latent + 1.0) / 2.0) *
                                 (AE.codebook.embeddings.max() - AE.codebook.embeddings.min())) + \
                                AE.codebook.embeddings.min()
            intra_latent_scaled = (((intra_latent + 1.0) / 2.0) *
                                   (AE.codebook.embeddings.max() - AE.codebook.embeddings.min())) + \
                                  AE.codebook.embeddings.min()
            
            pred_ct = AE.decode(gen_latent_scaled, quantize=True)
            target_ct = AE.decode(intra_latent_scaled, quantize=True)
            
            ct_mae = torch.mean(torch.abs(pred_ct - target_ct)).item()
            ct_mae_list.append(ct_mae)
            
            # 计算 SSIM 和 PSNR
            ct_metrics_tracker.update(pred_ct, target_ct)

            pbar.set_postfix({
                'latent_mae': f'{latent_mae:.4f}',
                'ct_mae': f'{ct_mae:.4f}'
            })

            # ======== 保存结果 ========
            if save_dir and (save_limit is None or idx < save_limit):
                # 获取 case name 和 level
                if isinstance(names, (list, tuple)):
                    name = names[0] if len(names) > 0 else f"sample_{idx}"
                else:
                    name = names
                
                # 获取 level 信息
                levels = batch.get('level', [f'L{level_idx[0].item() + 1}'])
                if isinstance(levels, (list, tuple)):
                    level = levels[0] if len(levels) > 0 else f"L{level_idx[0].item() + 1}"
                else:
                    level = levels
                
                # 创建 case 子目录: save_dir/case_name/
                case_dir = os.path.join(save_dir, name)
                os.makedirs(case_dir, exist_ok=True)
                
                # 保存预测 CT: save_dir/case_name/{level}_pred.nii.gz
                pred_path = os.path.join(case_dir, f"{level}_pred.nii.gz")
                tio.ScalarImage(tensor=pred_ct[0].detach().cpu()).save(pred_path)
                
                # 保存 GT CT: save_dir/case_name/{level}_gt.nii.gz
                gt_path = os.path.join(case_dir, f"{level}_gt.nii.gz")
                tio.ScalarImage(tensor=target_ct[0].detach().cpu()).save(gt_path)
                
                if logger:
                    logger.info(f"Saved {pred_path}")

    mean_latent_mae = float(np.mean(latent_mae_list)) if latent_mae_list else float("inf")
    mean_ct_mae = float(np.mean(ct_mae_list)) if ct_mae_list else float("inf")
    
    # 获取 SSIM 和 PSNR
    ct_metrics = ct_metrics_tracker.compute()
    
    return {
        'latent_mae': mean_latent_mae,
        'ct_mae': mean_ct_mae,
        'ct_ssim': ct_metrics['ssim'],
        'ct_psnr': ct_metrics['psnr'],
        'num_samples': len(latent_mae_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiFlowNet_Rec_Intra_Pre")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--ckpt", type=str, required=True, help="训练好的 checkpoint 路径 (.pt)")
    parser.add_argument("--val-files-names-path", type=str, help="验证集列表文件路径")
    parser.add_argument("--results-dir", type=str, help="输出/日志目录")
    parser.add_argument("--AE-ckpt", type=str, help="AutoEncoder checkpoint 路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--val-batch-size", type=int, default=1, help="验证 batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=None, help="只评估前 N 个样本")
    parser.add_argument("--save-samples", action="store_true", default=True, help="保存生成的 CT (默认开启)")
    parser.add_argument("--no-save-samples", dest="save_samples", action="store_false", help="不保存生成的 CT")
    parser.add_argument("--num-save-samples", type=int, default=5, help="最多保存 N 个样本")
    parser.add_argument("--no-ema", action="store_true", help="使用 model 而非 EMA")
    parser.add_argument("--enable-amp", action="store_true", default=False, help="启用混合精度")

    args = parser.parse_args()

    if not torch.cuda.is_available() and "cuda" in args.device:
        raise RuntimeError("未检测到 GPU，请使用 --device cpu")

    cfg = load_cfg(args)
    device = torch.device(args.device)
    amp = args.enable_amp

    # 输出目录
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    results_root = args.results_dir if args.results_dir else "./eval_results"
    eval_dir = os.path.join(results_root, f"eval_{ckpt_name}")
    samples_dir = os.path.join(eval_dir, "samples") if args.save_samples else None
    logger = create_logger(eval_dir)

    logger.info(f"Eval dir: {eval_dir}")
    logger.info(f"Device: {device} | AMP: {amp} | Use EMA: {not args.no_ema}")
    logger.info(f"Checkpoint: {args.ckpt}")

    # 随机种子
    seed = getattr(cfg.model, "global_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 构建模型
    model, diffusion, AE = build_model(cfg, device)
    ema_model = copy.deepcopy(model)
    eval_model = load_checkpoint(model, ema_model, args.ckpt, device, 
                                  use_ema=not args.no_ema, logger=logger)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 数据集
    val_dataset = Pre_Intra_Latent_Dataset(
        latent_root_dir=cfg.model.latent_root_dir,
        files_names_path=cfg.model.val_files_names_path,
        drr_roi_root=cfg.model.drr_roi_root,
    )
    # 限制验证集大小
    if args.max_samples:
        val_dataset.all_vertebral_level_path_pre = val_dataset.all_vertebral_level_path_pre[:args.max_samples]
        val_dataset.all_vertebral_level_path_intra = val_dataset.all_vertebral_level_path_intra[:args.max_samples]
    # pdb removed
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    logger.info(f"Validation size: {len(val_dataset)}, batch_size={args.val_batch_size}")

    # 评估
    eval_results = evaluate(
        eval_model,
        diffusion,
        AE,
        val_loader,
        device,
        amp,
        save_dir=samples_dir,
        save_limit=args.num_save_samples if args.save_samples else None,
        logger=logger,
    )

    # 保存指标
    metrics = {
        "checkpoint": args.ckpt,
        "latent_mae": eval_results['latent_mae'],
        "ct_mae": eval_results['ct_mae'],
        "ct_ssim": eval_results['ct_ssim'],
        "ct_psnr": eval_results['ct_psnr'],
        "num_samples": eval_results['num_samples'],
        "use_ema": not args.no_ema,
    }
    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("=" * 50)
    logger.info(f"Evaluation Results:")
    logger.info(f"  Latent MAE: {eval_results['latent_mae']:.4f}")
    logger.info(f"  CT MAE: {eval_results['ct_mae']:.4f}")
    logger.info(f"  CT SSIM: {eval_results['ct_ssim']:.4f}")
    logger.info(f"  CT PSNR: {eval_results['ct_psnr']:.2f} dB")
    logger.info(f"  Samples: {eval_results['num_samples']}")
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
