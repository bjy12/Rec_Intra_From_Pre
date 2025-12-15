"""
BiFlowNet 单 GPU 训练脚本（适配 Windows）
已修改：添加了 X-ray 和 CT Encoder 的配置定义
"""

import sys
import os
from types import SimpleNamespace # [新增] 引入 SimpleNamespace

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
import yaml
from ddpm.BiFlowNet_Rec import GaussianDiffusion
from ddpm.BiFlowNet_Rec import BiFlowNet_Pre_Intra 
from AutoEncoder.model.PatchVolume import patchvolumeAE
import torchio as tio
import copy
from torch.cuda.amp import autocast, GradScaler
import random
from omegaconf import OmegaConf
# from dataset.Singleres_dataset import Singleres_dataset
# from dataset.Singleres_dataset_ver_128 import Res_128_dataset
from dataset.Pre_Intra_dataset_ver_128 import Pre_Intra_Dataset_Ver_128
from torch.utils.data import DataLoader
from tqdm import tqdm

import pdb
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def dict_to_sns(d):
    """
    递归将字典转换为 SimpleNamespace，以便支持点号访问 (cfg.attribute)
    """
    if isinstance(d, dict):
        # 递归转换字典中的值
        return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})
    elif isinstance(d, list):
        # 如果列表中包含字典，也需要转换
        return [dict_to_sns(i) for i in d]
    else:
        return d

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
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def run_validation(
    *,
    model,
    ema_model,
    diffusion,
    AE,
    val_loader,
    device,
    amp,
    checkpoint_dir,
    samples_dir,
    scaler,
    opt,
    cfg_model,
    epoch,
    train_steps,
    logger,
    best_val_loss,
    tag: str = "epoch",
    max_gen_batches: int = 1,
):
    """
Run validation using the EMA model; save best checkpoint.
Evaluate generation (pre-CT + intra X-ray -> intra CT) using MAE only.
Returns updated best_val_loss and the mean generation MAE.
    """
    if val_loader is None:
        return best_val_loss, float("inf")

    ema_model.eval()
    gen_mae_list = []
    save_dir = os.path.join(samples_dir, "val_samples") if samples_dir else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    val_batches = 0

    with torch.no_grad():
        for ds_val in val_loader:
            pre_ct = ds_val['pre_ct']
            intra_ct = ds_val['intra_ct']
            projs = ds_val['projs']
            angles = ds_val['angles']
            projs_points = ds_val['proj_points']
            pre_latent = ds_val['pre_latent']
            intra_latent = ds_val['intra_latent']

            b = pre_ct.shape[0]

            z = intra_latent.to(device)
            projs = projs.to(device)
            projs_points = projs_points.to(device)
            pre_latent = pre_latent.to(device)

            condition_dict_val = {
                'projs': projs,
                'projs_points': projs_points,
                'pre_ct_latent': pre_latent
            }

            # ---- Full conditional generation check (lightweight; limited batches) ----
            if len(gen_mae_list) < max_gen_batches:
                z_sample = torch.randn_like(z, device=device)
                with autocast(enabled=amp):
                    gen_latent = diffusion.p_sample_loop(
                        ema_model, z_sample, condition_dict=condition_dict_val
                    )
                # scale back to codebook range then decode to CT volume
                gen_latent = (((gen_latent + 1.0) / 2.0) *
                              (AE.codebook.embeddings.max() - AE.codebook.embeddings.min())) + AE.codebook.embeddings.min()
                pred_ct = AE.decode(gen_latent, quantize=True)

                target_ct = intra_ct.to(device)
                mae_val = torch.mean(torch.abs(pred_ct - target_ct))
                gen_mae_list.append(mae_val.item())

                if save_dir:
                    name = ds_val.get('name', f"val_{val_batches}")
                    if isinstance(name, (list, tuple)):
                        name = name[0]
                    save_path = os.path.join(save_dir, f"{tag}_{name}_pred.nii.gz")
                    tio.ScalarImage(tensor=pred_ct.detach().cpu().squeeze(0)).save(save_path)
                    logger.info(f"Saved validation sample to {save_path}")

    if gen_mae_list:
        mean_gen_mae = float(np.mean(gen_mae_list))
        logger.info(f"({tag}) Gen MAE: {mean_gen_mae:.4f} (n={len(gen_mae_list)})")
    else:
        mean_gen_mae = float("inf")
        logger.warning(f"({tag}) No validation generations were produced; skipping metric.")

    if mean_gen_mae < best_val_loss:
        best_val_loss = mean_gen_mae
        best_checkpoint = {
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "scaler": scaler.state_dict(),
            "opt": opt.state_dict(),
            "args": cfg_model,
            "epoch": epoch,
            "train_steps": train_steps,
            "best_val_loss": best_val_loss
        }
        best_ckpt_path = f"{checkpoint_dir}/best.pt"
        torch.save(best_checkpoint, best_ckpt_path)
        logger.info(f"New best checkpoint saved to {best_ckpt_path}")

    model.train()
    return best_val_loss, mean_gen_mae


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(cfg):
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
    torch.manual_seed(cfg.model.global_seed)
    torch.cuda.manual_seed_all(cfg.model.global_seed)
    random.seed(cfg.model.global_seed)
    np.random.seed(cfg.model.global_seed)
    
    # Setup experiment folder:
    os.makedirs(cfg.model.results_dir, exist_ok=True)
    if cfg.model.ckpt == None:
        experiment_index = len(glob(f"{cfg.model.results_dir}/*"))
        model_string_name = cfg.model.model 
        experiment_dir = f"{cfg.model.results_dir}/{experiment_index:03d}-{model_string_name}"
    else:
        experiment_dir = os.path.dirname(os.path.dirname(cfg.model.ckpt))
    
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    samples_dir = f"{experiment_dir}/samples" 
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(f"Using device: {device}")

    # pdb.set_trace()
    # Create model:
    model = BiFlowNet_Pre_Intra(
            dim=cfg.model.model_dim,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.volume_channels,
            init_kernel_size=3,
            cond_classes=None,
            res_condition=False,
            learn_sigma=False,
            use_sparse_linear_attn=cfg.model.use_attn,
            vq_size=cfg.model.vq_size,
            num_mid_DiT = cfg.model.num_dit,
            patch_size = cfg.model.patch_size,
            # [修改] 传入上面定义的配置对象
            cfg_xray_encoder=cfg.cfg_xray_encoder,
            cfg_ct_encoder=cfg.cfg_ct_encoder,
            condition_channels=cfg.model.condition_channels
        ).to(device)
    
    diffusion = GaussianDiffusion(
        channels=cfg.model.volume_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    ).to(device)
    logger.info(f"*****************Diffusion Model loaded  Successfully*****************")
    #pdb.set_trace()
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model)
    update_ema_every = 10
    step_start_ema = 2000
    
    amp = cfg.model.enable_amp
    scaler = GradScaler(enabled=amp)
    
    if cfg.model.AE_ckpt:
        # 加载 VQGAN/AutoEncoder，这里假设 patchvolumeAE 已经正确定义
        # 如果 patchvolumeAE 需要配置，请确保这里能正确加载
        AE = patchvolumeAE.load_from_checkpoint(cfg.model.AE_ckpt).to(device)
        AE.eval()
    else:
        raise NotImplementedError("AutoEncoder checkpoint is required!")
     
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"***************** AE Model Loaded Successfully *****************")

    # Setup optimizer:
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Load checkpoint if provided:
    if cfg.model.ckpt:
        checkpoint = torch.load(cfg.model.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)
        ema_model.load_state_dict(checkpoint['ema'], strict=True)
        scaler.load_state_dict(checkpoint['scaler'])
        opt.load_state_dict(checkpoint['opt'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'train_steps' in checkpoint:
            train_steps = checkpoint['train_steps']
        del checkpoint 
        logger.info(f'Resuming from checkpoint: {cfg.model.ckpt}')
        logger.info(f'Resuming from step: {train_steps}, epoch: {start_epoch}')
    
    # Setup data:
    # Setup training dataset
    dataset = Pre_Intra_Dataset_Ver_128(
        root_dir=cfg.model.data_path,
        files_names_path=cfg.model.train_files_names_path,
        geo_config_path=cfg.model.geo_config_path)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.model.batch_size, 
        num_workers=cfg.model.num_workers,
        shuffle=True,
    )
    
    # Setup validation dataset
    val_loader = None
    if getattr(cfg.model, "val_files_names_path", None):
        val_dataset = Pre_Intra_Dataset_Ver_128(
            root_dir=cfg.model.data_path,
            files_names_path=cfg.model.val_files_names_path,
            geo_config_path=cfg.model.geo_config_path)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )
        logger.info(f"Validation size: {len(val_dataset)}")
    else:
        logger.warning("No val_files_names_path provided; validation will be skipped.")

    
    logger.info(f"Train Dataset size: {len(dataset)}")
    logger.info(f"Train Batch size: {cfg.model.batch_size}")
    logger.info(f"Train Number of batches per epoch: {len(loader)}")
    model.train()  
    ema_model.eval() 

    log_steps = 0
    running_loss = 0
    start_time = time()
    best_val_loss = float("inf")
    # optional mid-epoch validation frequency
    val_every_steps = getattr(cfg.model, "val_every_steps", None)
    
    #pdb.set_trace()
    logger.info(f"Training for {cfg.model.epochs} epochs...")
    logger.info(f"Checkpoint every {cfg.model.ckpt_every} steps")
    logger.info(f"Log every {cfg.model.log_every} steps")
    #pdb.set_trace()
    epoch_pbar = tqdm(range(start_epoch, cfg.model.epochs), desc="Epochs", position=0, leave=True)
    
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}/{cfg.model.epochs-1}")
        logger.info(f"Beginning epoch {epoch}...")
        
        batch_pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}", 
                         position=1, leave=False, ncols=120)
        #pdb.set_trace()
        for batch_idx, ds in batch_pbar:
            # 数据解包
            pre_ct = ds['pre_ct'] # B C H W D 
            intra_ct = ds['intra_ct'] # B C H W D 
            # name = ds['name'] # 暂时用不到 name
            projs = ds['projs'] # B N_views C H W
            # angles = ds['angles']
            pre_latent = ds['pre_latent'] # B l_c l_h l_w l_d 
            intra_latent = ds['intra_latent'] 
            projs_points = ds['proj_points'] # B N_views 3

            #pdb.set_trace()
            b = pre_ct.shape[0]
            
            # 移动到 GPU
            z = intra_latent.to(device) # GT (Target) # 
            projs = projs.to(device)
            # angles 通常对应 proj_points (几何参数)
            #angles = angles.to(device) 
            # pre_ct = pre_ct.to(device) # 如果不需要原始 CT 像素，可以不移
            pre_latent = pre_latent.to(device)
            projs_points = projs_points.to(device)
            # 构建 Condition Dict
            #pdb.set_trace()
            condition_dict = {
                'projs': projs,
                'projs_points': projs_points, 
                'pre_ct_latent': pre_latent  
            }
            #pdb.set_trace()
            with autocast(enabled=amp):
                t = torch.randint(0, diffusion.num_timesteps, (b,), device=device)
                # 传入 condition_dict
                loss = diffusion.p_losses(model, z, t, condition_dict=condition_dict)
                scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            opt.zero_grad()          

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % update_ema_every == 0:
                if train_steps < step_start_ema:
                    ema_model.load_state_dict(model.state_dict(), strict=True)
                else:
                    ema.update_model_average(ema_model, model)

            current_lr = opt.state_dict()['param_groups'][0]['lr']
            avg_loss = running_loss / log_steps if log_steps > 0 else loss.item()
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'step': train_steps,
                'lr': f'{current_lr:.6f}'
            })

            if train_steps % cfg.model.log_every == 0:
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
            # Save checkpoint & Sample
            if train_steps % cfg.model.ckpt_every == 0 and train_steps > 0:
                # ... (保持原有的保存和采样逻辑不变) ...
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "scaler": scaler.state_dict(),
                    "opt": opt.state_dict(),
                    "args": cfg.model,
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
                # 1. 取当前 batch 的 condition
                # 2. 传入 p_sample_loop
                #pdb.set_trace()
                with torch.no_grad():
                    milestone = train_steps // cfg.model.ckpt_every
                    # 使用当前训练 batch 的 condition (或者最好从 val set 取)
                    # z_sample 是纯噪声
                    z_sample = torch.randn(1, cfg.model.volume_channels, 
                                         cfg.model.resolution[0], cfg.model.resolution[1], cfg.model.resolution[2], 
                                         device=device)
                    
                    # 构造单样本 condition (取 batch 的第一个)
                    #pdb.set_trace()
                    sample_cond_dict = {
                        'projs': projs[0:1],
                        'projs_points': projs_points[0:1],
                        'pre_ct_latent': pre_latent[0:1]
                    }
                    name = ds['name']
                    # 这里的 p_sample_loop 需要支持传入 condition_dict
                    # 请确保 GaussianDiffusion.p_sample_loop 已经修改支持 **kwargs 或显式参数
                    samples = diffusion.p_sample_loop(
                        ema_model, z_sample, condition_dict=sample_cond_dict # 假设你修改后的接口用 hint 接收字典
                    )
                    # pdb.set_trace()
                    # 解码保存... (保持不变)
                    samples = (((samples + 1.0) / 2.0) * (AE.codebook.embeddings.max() -
                                AE.codebook.embeddings.min())) + AE.codebook.embeddings.min()
                    torch.cuda.empty_cache()
                    volume = AE.decode(samples, quantize=True)
                    volume_path = os.path.join(samples_dir, f'{milestone}_{name}.nii.gz')
                    volume = volume.detach().squeeze(0).cpu()
                    tio.ScalarImage(tensor=volume).save(volume_path)
                    logger.info(f"Generated sample: {volume_path}")
                torch.cuda.empty_cache()
            # Mid-epoch validation at fixed step intervals (if configured)
            if val_loader is not None and val_every_steps and train_steps % val_every_steps == 0:
                best_val_loss, _ = run_validation(
                    model=model,
                    ema_model=ema_model,
                    diffusion=diffusion,
                    AE=AE,
                    val_loader=val_loader,
                    device=device,
                    amp=amp,
                    checkpoint_dir=checkpoint_dir,
                    samples_dir=samples_dir,
                    scaler=scaler,
                    opt=opt,
                    cfg_model=cfg.model,
                    epoch=epoch,
                    train_steps=train_steps,
                    logger=logger,
                    best_val_loss=best_val_loss,
                    tag=f"step={train_steps:07d}",
                    max_gen_batches=getattr(cfg.model, "val_gen_batches", 1),
                )
        batch_pbar.close()
        # ----------------- Validation ----------------- #
        if val_loader is not None and not val_every_steps:
            best_val_loss, _ = run_validation(
                model=model,
                ema_model=ema_model,
                diffusion=diffusion,
                AE=AE,
                val_loader=val_loader,
                device=device,
                amp=amp,
                checkpoint_dir=checkpoint_dir,
                samples_dir=samples_dir,
                scaler=scaler,
                opt=opt,
                cfg_model=cfg.model,
                epoch=epoch,
                train_steps=train_steps,
                logger=logger,
                best_val_loss=best_val_loss,
                tag=f"epoch={epoch}",
                max_gen_batches=getattr(cfg.model, "val_gen_batches", 1),
            )
    model.eval()
    logger.info("Training completed!")
    logger.info(f"Final checkpoint saved at: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiFlowNet (Single GPU version)")
    # 先读取 --config，再用 YAML 中的键作为默认值
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file to override defaults")
    parser.add_argument("--data-path", type=str, required=False,
                        help="Path to latent dataset root directory")
    parser.add_argument("--files-names-path", type=str, required=False,
                        help="Path to file containing list of file names (one per line)")
    parser.add_argument("--val-files-names-path", type=str, default=None,
                        help="Path to validation file list (one per line)")
    parser.add_argument("--results-dir", type=str, required=False,
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
    parser.add_argument("--AE-ckpt", type=str, required=False,
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
    parser.add_argument("--val-batch-size", type=int, default=None,
                        help="Validation batch size (defaults to train batch size)")
    parser.add_argument("--val-every-steps", type=int, default=None,
                        help="Run validation every N training steps (None to disable)")
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
    
    # [新增] condition channels 参数
    parser.add_argument("--condition-channels", type=int, default=128,
                        help="Number of channels for condition feature")
    #pdb.set_trace()
    args = parser.parse_args()
    if args.config is not None:
        cfg = OmegaConf.load(args.config)
        main(cfg)
    else:
        main(args)