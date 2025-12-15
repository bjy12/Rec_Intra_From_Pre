"""
BiFlowNet 单 GPU 训练脚本（适配 Windows）
原脚本使用 DDP，Windows 不支持 NCCL，此版本适配单 GPU 训练
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
from collections import OrderedDict
from glob import glob
from time import time
import argparse
import logging
from ddpm.BiFlowNet import  GaussianDiffusion
from ddpm.BiFlowNet import BiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE
import torchio as tio
import copy
from torch.cuda.amp import autocast, GradScaler
import random
#from dataset.Singleres_dataset import Singleres_dataset
from dataset.Singleres_dataset_ver_128 import Res_128_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains BiFlowNet model (Single GPU version for Windows)
    """
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    start_epoch = 0
    train_steps = 0
    log_steps = 0
    running_loss = 0
    
    device = torch.device("cuda:0")
    torch.manual_seed(args.global_seed)
    torch.cuda.manual_seed_all(args.global_seed)
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
    
    # Setup experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    if args.ckpt == None:
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model 
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    else:
        experiment_dir = os.path.dirname(os.path.dirname(args.ckpt))
    
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    samples_dir = f"{experiment_dir}/samples" 
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(f"Using device: {device}")

    # Create model:
    model = BiFlowNet(
            dim=args.model_dim,
            dim_mults=args.dim_mults,
            channels=args.volume_channels,
            init_kernel_size=3,
            cond_classes=args.num_classes,
            learn_sigma=False,
            use_sparse_linear_attn=args.use_attn,
            vq_size=args.vq_size,
            num_mid_DiT = args.num_dit,
            patch_size = args.patch_size
        ).to(device)
    
    diffusion = GaussianDiffusion(
        channels=args.volume_channels,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
    ).to(device)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model)
    update_ema_every = 10
    step_start_ema = 2000
    
    amp = args.enable_amp
    scaler = GradScaler(enabled=amp)
    
    if args.AE_ckpt:
        AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt).to(device)
        AE.eval()
    else:
        raise NotImplementedError("AutoEncoder checkpoint is required!")

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer:
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Load checkpoint if provided:
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)
        ema_model.load_state_dict(checkpoint['ema'], strict=True)
        scaler.load_state_dict(checkpoint['scaler'])
        opt.load_state_dict(checkpoint['opt'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'train_steps' in checkpoint:
            train_steps = checkpoint['train_steps']
        del checkpoint 
        logger.info(f'Resuming from checkpoint: {args.ckpt}')
        logger.info(f'Resuming from step: {train_steps}, epoch: {start_epoch}')
    
    # Setup data:
    dataset = Res_128_dataset(
        root_dir=args.data_path,
        files_names_path=args.files_names_path,
        resolution=args.resolution,
        generate_latents=False
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=True,
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of batches per epoch: {len(loader)}")

    model.train()  
    ema_model.eval() 

    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    logger.info(f"Checkpoint every {args.ckpt_every} steps")
    logger.info(f"Log every {args.log_every} steps")
    
    # 外层 epoch 进度条
    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Epochs", position=0, leave=True)
    
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}/{args.epochs-1}")
        logger.info(f"Beginning epoch {epoch}...")
        
        # 内层 batch 进度条
        batch_pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}", 
                         position=1, leave=False, ncols=120)
        
        for batch_idx, (z, y, res) in batch_pbar:
            b = z.shape[0]
            z = z.to(device)
            y = y.to(device)
            res = res.to(device)
            
            with autocast(enabled=amp):
                t = torch.randint(0, diffusion.num_timesteps, (b,), device=device)
                loss = diffusion.p_losses(model, z, t, y=y, res=res)
                scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            opt.zero_grad()          

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            # Update EMA:
            if train_steps % update_ema_every == 0:
                if train_steps < step_start_ema:
                    ema_model.load_state_dict(model.state_dict(), strict=True)
                else:
                    ema.update_model_average(ema_model, model)

            # 更新进度条显示
            current_lr = opt.state_dict()['param_groups'][0]['lr']
            avg_loss = running_loss / log_steps if log_steps > 0 else loss.item()
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'step': train_steps,
                'lr': f'{current_lr:.6f}'
            })

            # Logging:
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                          f"Train Steps/Sec: {steps_per_sec:.2f}, "
                          f"LR: {opt.state_dict()['param_groups'][0]['lr']:.6f}")
                running_loss = 0
                log_steps = 0
                start_time = time()
            
            # Save checkpoint and generate samples:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "scaler": scaler.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "train_steps": train_steps
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                ckpt_avg_loss = running_loss / log_steps if log_steps > 0 else loss.item()
                ckpt_lr = opt.state_dict()['param_groups'][0]['lr']
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{ckpt_avg_loss:.4f}',
                    'step': train_steps,
                    'lr': f'{ckpt_lr:.6f}',
                    'ckpt': 'saved'
                })
                
                # Keep only last 6 checkpoints:
                checkpoints = sorted(glob(f"{checkpoint_dir}/*.pt"))
                if len(checkpoints) > 6:
                    os.remove(checkpoints[0])
                
                # Generate sample:
                with torch.no_grad():
                    milestone = train_steps // args.ckpt_every
                    # 对于单类别数据集，固定使用类别 0
                    if args.num_classes == 1:
                        cls_num = 0
                    else:
                        cls_num = np.random.choice(list(range(0, args.num_classes)))
                    volume_size = args.resolution
                    z_sample = torch.randn(1, args.volume_channels, 
                                         volume_size[0], volume_size[1], volume_size[2], 
                                         device=device)
                    y_sample = torch.tensor([cls_num], device=device)
                    res_sample = torch.tensor(volume_size, device=device) / 64.0
                    
                    samples = diffusion.p_sample_loop(
                        ema_model, z_sample, y=y_sample, res=res_sample
                    )
                    samples = (((samples + 1.0) / 2.0) * (AE.codebook.embeddings.max() -
                                AE.codebook.embeddings.min())) + AE.codebook.embeddings.min()
                    torch.cuda.empty_cache()

                    volume = AE.decode(samples, quantize=True)
                    volume_path = os.path.join(samples_dir, f'{milestone}_{cls_num}.nii.gz') 
                    volume = volume.detach().squeeze(0).cpu()
                    volume = volume.transpose(1,3).transpose(1,2)
                    tio.ScalarImage(tensor=volume).save(volume_path)
                    logger.info(f"Generated sample: {volume_path}")
                torch.cuda.empty_cache()
        # Epoch 结束，关闭内层进度条
        batch_pbar.close()
    
    model.eval()
    logger.info("Training completed!")
    logger.info(f"Final checkpoint saved at: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiFlowNet (Single GPU version)")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to latent dataset root directory")
    parser.add_argument("--files-names-path", type=str, required=True,
                        help="Path to file containing list of file names (one per line)")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--loss-type", type=str, default='l1',
                        choices=['l1', 'l2'],
                        help="Loss type")
    parser.add_argument("--volume-channels", type=int, default=8,
                        help="Number of channels in latent space")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--model-dim", type=int, default=72,
                        help="Model dimension")
    parser.add_argument("--dim-mults", nargs='+', type=int, default=[1,1,2,4,8],
                        help="U-Net channel multipliers")
    parser.add_argument("--use-attn", nargs='+', type=int, default=[0,0,0,1,1],
                        help="Use attention at each layer")
    parser.add_argument("--patch-size", type=int, default=1,
                        help="Patch size")
    parser.add_argument("--num-dit", type=int, default=1,
                        help="Number of DiT blocks")
    parser.add_argument("--enable_amp", action='store_true', default=True,
                        help="Enable mixed precision training")
    parser.add_argument("--model", type=str, default="BiFlowNet",
                        help="Model name")
    parser.add_argument("--AE-ckpt", type=str, required=True,
                        help="Path to AutoEncoder checkpoint")
    parser.add_argument("--num-classes", type=int, default=7,
                        help="Number of classes")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--global-seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument('--resolution', nargs='+', type=int, default=[32, 32, 32],
                        help="Latent space resolution")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log every N steps")
    parser.add_argument("--ckpt-every", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--vq-size", type=int, default=64,
                        help="VQ size")
    
    args = parser.parse_args()
    main(args)

