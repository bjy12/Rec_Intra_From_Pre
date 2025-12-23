"""
Stage 1 Training Script: Multi-Task Condition Branch Pre-training (V2)

适配新的 dataset 格式 (无 proj_points)

训练多任务条件分支网络：
- Task 1 (Registration): 预测 6-DoF 刚体变换参数
- Task 2 (Generation): 预测术中 CT

Usage:
    python train/train_multitask_condition.py --config config/multitask_condition.yaml
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import argparse
import logging
import copy
import yaml
from glob import glob
from time import time
from tqdm import tqdm
import torchio as tio

from dataset.Pre_Intra_Final_Dataset import Pre_Intra_Final_Dataset
from ddpm.model.multitask_condition_model import MultiTaskConditionBranchV3


class EMA:
    """Exponential Moving Average"""
    def __init__(self, beta: float = 0.995):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for ma_params, current_params in zip(ma_model.parameters(), current_model.parameters()):
            ma_params.data = self.beta * ma_params.data + (1 - self.beta) * current_params.data


def create_logger(logging_dir: str):
    """Create logger with file and console handlers."""
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt")
        ]
    )
    return logging.getLogger(__name__)


def main(args):
    """Main training loop."""
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    assert torch.cuda.is_available(), "Training requires GPU."
    
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Create experiment directory
    os.makedirs(args.results_dir, exist_ok=True)
    if args.ckpt is None:
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-MultiTaskCondition"
    else:
        experiment_dir = os.path.dirname(os.path.dirname(args.ckpt))
    
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    samples_dir = f"{experiment_dir}/samples"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Device: {device}")
    
    # Create V3 model
    model = MultiTaskConditionBranchV3(
        in_channels=1,
        base_channels=args.base_channels,
        feat_channels=args.feat_channels,
        num_attention_heads=args.num_attention_heads,
        max_rotation_deg=args.max_rotation_deg,
        max_translation_voxels=args.max_translation_voxels,
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # EMA
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Mixed precision
    scaler = GradScaler(enabled=args.enable_amp)
    
    # Resume from checkpoint
    start_epoch = 0
    train_steps = 0
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        train_steps = checkpoint.get('train_steps', 0)
        logger.info(f"Resumed from {args.ckpt}, epoch {start_epoch}, step {train_steps}")
    
    # Dataset
    train_dataset = Pre_Intra_Final_Dataset(
        root_dir=args.data_path,
        files_names_path=args.train_files_path,
        geo_config_path=args.geo_config_path,
        max_rotation_deg=args.max_rotation_deg,
        max_translation_voxels=args.max_translation_voxels,
        apply_perturbation=True,
        window_min=args.window_min,
        window_max=args.window_max,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    
    # Validation dataset (optional)
    val_loader = None
    if args.val_files_path is not None:
        val_dataset = Pre_Intra_Final_Dataset(
            root_dir=args.data_path,
            files_names_path=args.val_files_path,
            geo_config_path=args.geo_config_path,
            apply_perturbation=False,
            window_min=args.window_min,
            window_max=args.window_max,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Loss weights
    lambda_reg = args.lambda_reg
    lambda_gen = args.lambda_gen
    logger.info(f"Loss weights: lambda_reg={lambda_reg}, lambda_gen={lambda_gen}")
    
    # Training loop
    model.train()
    running_loss = 0.0
    running_reg_loss = 0.0
    running_gen_loss = 0.0
    log_steps = 0
    start_time = time()
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", position=1, leave=False, ncols=120)
        
        for batch_idx, batch in enumerate(batch_pbar):
            # Data (updated to match new dataset format)
            perturbed_ct = batch['perturbed_ct'].to(device)      # [B, 1, D, H, W]
            pre_aligned = batch['pre_aligned'].to(device)         # [B, 1, D, H, W] - Registration GT
            intra_ct = batch['intra_ct'].to(device)               # [B, 1, D, H, W] - Generation GT
            drr_images = batch['drr_images'].to(device)           # [B, 2, 1, H, W]
            transform_params_gt = batch['transform_params'].to(device)  # [B, 6]
            
            optimizer.zero_grad()
            
            with autocast(enabled=args.enable_amp):
                # Forward (no proj_points needed)
                outputs = model(perturbed_ct, drr_images)
                transform_pred = outputs['transform_pred']    # [B, 6]
                intra_ct_pred = outputs['intra_ct_pred']      # [B, 1, D, H, W]
                
                # Registration loss (L1 on transform parameters)
                loss_reg = F.l1_loss(transform_pred, transform_params_gt)
                
                # Generation loss (L1 on CT reconstruction)
                loss_gen = F.l1_loss(intra_ct_pred, intra_ct)
                
                # Total loss
                loss = lambda_reg * loss_reg + lambda_gen * loss_gen
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            train_steps += 1
            if train_steps % 10 == 0:
                if train_steps < 2000:
                    ema_model.load_state_dict(model.state_dict())
                else:
                    ema.update_model_average(ema_model, model)
            
            # Logging
            running_loss += loss.item()
            running_reg_loss += loss_reg.item()
            running_gen_loss += loss_gen.item()
            log_steps += 1
            
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reg': f'{loss_reg.item():.4f}',
                'gen': f'{loss_gen.item():.4f}',
                'step': train_steps
            })
            
            if train_steps % args.log_every == 0:
                avg_loss = running_loss / log_steps
                avg_reg = running_reg_loss / log_steps
                avg_gen = running_gen_loss / log_steps
                elapsed = time() - start_time
                steps_per_sec = log_steps / elapsed
                lr = optimizer.param_groups[0]['lr']
                
                logger.info(
                    f"[Step {train_steps:07d}] Loss: {avg_loss:.4f} "
                    f"(reg={avg_reg:.4f}, gen={avg_gen:.4f}), "
                    f"LR: {lr:.6f}, Steps/sec: {steps_per_sec:.2f}"
                )
                
                running_loss = 0.0
                running_reg_loss = 0.0
                running_gen_loss = 0.0
                log_steps = 0
                start_time = time()
            
            # Save checkpoint
            if train_steps % args.ckpt_every == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'ema': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'train_steps': train_steps,
                    'args': vars(args)
                }
                ckpt_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")
                
                # Keep only last N checkpoints
                ckpts = sorted(glob(f"{checkpoint_dir}/*.pt"))
                while len(ckpts) > args.max_ckpts:
                    os.remove(ckpts.pop(0))
                
                # Save sample
                with torch.no_grad():
                    sample_pred = intra_ct_pred[0:1].detach().cpu()
                    sample_gt = intra_ct[0:1].detach().cpu()
                    sample_perturbed = perturbed_ct[0:1].detach().cpu()
                    
                    tio.ScalarImage(tensor=sample_pred.squeeze(0)).save(
                        f"{samples_dir}/step{train_steps:07d}_pred.nii.gz"
                    )
                    tio.ScalarImage(tensor=sample_gt.squeeze(0)).save(
                        f"{samples_dir}/step{train_steps:07d}_gt.nii.gz"
                    )
                    tio.ScalarImage(tensor=sample_perturbed.squeeze(0)).save(
                        f"{samples_dir}/step{train_steps:07d}_perturbed.nii.gz"
                    )
                    logger.info(f"Saved samples at step {train_steps}")
        
        # End of epoch
        scheduler.step()
        
        # Validation
        if val_loader is not None and (epoch + 1) % args.val_every_epochs == 0:
            model.eval()
            val_reg_loss = 0.0
            val_gen_loss = 0.0
            n_val = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    perturbed_ct = val_batch['perturbed_ct'].to(device)
                    intra_ct = val_batch['intra_ct'].to(device)
                    drr_images = val_batch['drr_images'].to(device)
                    transform_params_gt = val_batch['transform_params'].to(device)
                    
                    outputs = ema_model(perturbed_ct, drr_images)
                    val_reg_loss += F.l1_loss(outputs['transform_pred'], transform_params_gt).item()
                    val_gen_loss += F.l1_loss(outputs['intra_ct_pred'], intra_ct).item()
                    n_val += 1
            
            val_reg_loss /= max(n_val, 1)
            val_gen_loss /= max(n_val, 1)
            logger.info(f"[Epoch {epoch}] Validation - Reg Loss: {val_reg_loss:.4f}, Gen Loss: {val_gen_loss:.4f}")
            model.train()
    
    # Final save
    final_ckpt = {
        'model': model.state_dict(),
        'ema': ema_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': args.epochs - 1,
        'train_steps': train_steps,
        'args': vars(args)
    }
    torch.save(final_ckpt, f"{checkpoint_dir}/final.pt")
    logger.info(f"Training complete! Final checkpoint saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Task Condition Branch V3")
    
    # Paths
    parser.add_argument("--data-path", type=str, required=True, help="Path to final_dataset directory")
    parser.add_argument("--train-files-path", type=str, required=True, help="Path to train files list")
    parser.add_argument("--val-files-path", type=str, default=None, help="Path to val files list")
    parser.add_argument("--geo-config-path", type=str, required=True, help="Path to geometry config yaml")
    parser.add_argument("--results-dir", type=str, default="./results/multitask_condition_v3", help="Results directory")
    parser.add_argument("--ckpt", type=str, default=None, help="Resume from checkpoint")
    
    # Data augmentation and normalization
    parser.add_argument("--max-rotation-deg", type=float, default=20.0, help="Max rotation in degrees")
    parser.add_argument("--max-translation-voxels", type=float, default=10.0, help="Max translation in voxels")
    parser.add_argument("--window-min", type=float, default=-250, help="CT window min")
    parser.add_argument("--window-max", type=float, default=2000, help="CT window max")
    
    # Model V3 parameters
    parser.add_argument("--base-channels", type=int, default=32, help="Base channels for encoders")
    parser.add_argument("--feat-channels", type=int, default=128, help="Feature channels after encoding")
    parser.add_argument("--num-attention-heads", type=int, default=4, help="Number of attention heads")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (1 recommended for 128^3)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-reg", type=float, default=1.0, help="Registration loss weight")
    parser.add_argument("--lambda-gen", type=float, default=1.0, help="Generation loss weight")
    parser.add_argument("--enable-amp", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Logging
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--max-ckpts", type=int, default=5)
    parser.add_argument("--val-every-epochs", type=int, default=5)
    
    args = parser.parse_args()
    main(args)

