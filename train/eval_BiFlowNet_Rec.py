"""
BiFlowNet 单 GPU 验证脚本

用途：加载训练好的 BiFlowNet_Rec 检查点，运行验证集并计算 MAE，
可选择保存生成的 CT 体数据。
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

# 项目内模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from ddpm.BiFlowNet_Rec import GaussianDiffusion, BiFlowNet_Pre_Intra
from AutoEncoder.model.PatchVolume import patchvolumeAE
from dataset.Pre_Intra_dataset_ver_128 import Pre_Intra_Dataset_Ver_128


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

    if args.data_path:
        cfg.model.data_path = args.data_path
    if args.val_files_names_path:
        cfg.model.val_files_names_path = args.val_files_names_path
    if args.geo_config_path:
        cfg.model.geo_config_path = args.geo_config_path
    if args.results_dir:
        cfg.model.results_dir = args.results_dir
    if args.ckpt:
        cfg.model.ckpt = args.ckpt
    if args.AE_ckpt:
        cfg.model.AE_ckpt = args.AE_ckpt
    if args.val_batch_size is not None:
        cfg.model.val_batch_size = args.val_batch_size
    if args.condition_channels is not None:
        cfg.model.condition_channels = args.condition_channels
    if args.enable_amp is not None:
        cfg.model.enable_amp = args.enable_amp

    return cfg


class AttrDict(dict):
    """支持 attr 与 key 访问的字典，兼容 xray_encoder/ct_encoder 不同写法。"""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def to_attrdict(obj):
    """递归将 SimpleNamespace / OmegaConf 容器转换为 AttrDict。"""
    try:
        from omegaconf import OmegaConf as _OC  # 局部导入避免循环
        if _OC.is_config(obj):
            obj = _OC.to_container(obj, resolve=True)
    except Exception:
        pass
    if isinstance(obj, SimpleNamespace):
        obj = vars(obj)
    if isinstance(obj, dict):
        return AttrDict({k: to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_attrdict(i) for i in obj]
    return obj


def build_model(cfg, device):
    # condition_model / encoders 需要同时支持 attr 与下标访问，使用 AttrDict 包装
    cfg_xray_encoder = to_attrdict(cfg.cfg_xray_encoder)
    cfg_ct_encoder = to_attrdict(cfg.cfg_ct_encoder)

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
        num_mid_DiT=cfg.model.num_dit,
        patch_size=cfg.model.patch_size,
        cfg_xray_encoder=cfg_xray_encoder,
        cfg_ct_encoder=cfg_ct_encoder,
        condition_channels=cfg.model.condition_channels,
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
    if ckpt_path is None:
        raise ValueError("必须提供 --ckpt 路径以进行验证。")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    if "ema" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema"], strict=True)
    else:
        # 当没有 EMA 权重时，直接复制模型参数
        ema_model.load_state_dict(model.state_dict(), strict=True)

    eval_model = ema_model if use_ema and "ema" in checkpoint else model
    if logger:
        logger.info(
            f"Loaded checkpoint from {ckpt_path} | use_ema={use_ema} "
            f"(epoch={checkpoint.get('epoch', 'NA')}, step={checkpoint.get('train_steps', 'NA')})"
        )
    return eval_model


def evaluate(model, diffusion, AE, val_loader, device, amp, save_dir=None, save_limit=None, logger=None):
    model.eval()
    mae_list = []
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            projs = batch["projs"].to(device)
            pre_latent = batch["pre_latent"].to(device)
            projs_points = batch["proj_points"].to(device)
            target_latent = batch["intra_latent"].to(device)
            target_ct = batch["intra_ct"].to(device)

            cond = {
                "projs": projs,
                "projs_points": projs_points,
                "pre_ct_latent": pre_latent,
            }

            z_sample = torch.randn_like(target_latent, device=device)
            with autocast(enabled=amp):
                gen_latent = diffusion.p_sample_loop(model, z_sample, condition_dict=cond)

            # 还原到 VQ codebook 范围再解码
            gen_latent = (((gen_latent + 1.0) / 2.0) *
                          (AE.codebook.embeddings.max() - AE.codebook.embeddings.min())) + AE.codebook.embeddings.min()
            pred_ct = AE.decode(gen_latent, quantize=True)

            mae = torch.mean(torch.abs(pred_ct - target_ct)).item()
            mae_list.append(mae)

            if save_dir and (save_limit is None or len(mae_list) <= save_limit):
                names = batch.get("name", [f"sample_{idx}"])
                if isinstance(names, str):
                    names = [names] * pred_ct.shape[0]
                for b_idx in range(pred_ct.shape[0]):
                    name_str = names[b_idx] if b_idx < len(names) else f"{idx}_{b_idx}"
                    save_path = os.path.join(save_dir, f"{name_str}_pred.nii.gz")
                    tio.ScalarImage(tensor=pred_ct[b_idx].detach().cpu()).save(save_path)
                    if logger:
                        logger.info(f"Saved {save_path}")

    mean_mae = float(np.mean(mae_list)) if mae_list else float("inf")
    return mean_mae, len(mae_list)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiFlowNet (Single GPU)")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--ckpt", type=str, required=True, help="训练好的 checkpoint 路径 (.pt)")
    parser.add_argument("--data-path", type=str, help="数据根目录，覆盖配置中的 model.data_path")
    parser.add_argument("--val-files-names-path", type=str, help="验证集列表文件路径")
    parser.add_argument("--geo-config-path", type=str, help="几何配置 YAML 路径")
    parser.add_argument("--results-dir", type=str, help="输出/日志目录")
    parser.add_argument("--AE-ckpt", type=str, help="AutoEncoder checkpoint 路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备，如 cuda:0 或 cpu")
    parser.add_argument("--val-batch-size", type=int, default=None, help="验证 batch size（默认 1）")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers 数量")
    parser.add_argument("--max-val-batches", type=int, default=None, help="仅跑前 N 个 batch 以快速验证")
    parser.add_argument("--save-samples", action="store_true", help="保存生成的 CT 体数据")
    parser.add_argument("--num-save-samples", type=int, default=4, help="最多保存前 N 个 batch 的结果")
    parser.add_argument("--no-ema", action="store_true", help="使用模型权重而非 EMA 权重")
    parser.add_argument("--enable-amp", dest="enable_amp", action="store_true", help="启用混合精度")
    parser.add_argument("--disable-amp", dest="enable_amp", action="store_false", help="禁用混合精度")
    parser.add_argument("--condition-channels", type=int, default=None, help="覆盖条件通道数")
    parser.set_defaults(enable_amp=None)

    args = parser.parse_args()

    if not torch.cuda.is_available() and "cuda" in args.device:
        raise RuntimeError("未检测到可用 GPU，请设置 --device cpu 或检查 CUDA 环境。")

    cfg = load_cfg(args)
    device = torch.device(args.device)
    amp = cfg.model.enable_amp if args.enable_amp is None else args.enable_amp

    # 基本检查
    required_fields = ["data_path", "val_files_names_path", "geo_config_path", "ckpt"]
    for field in required_fields:
        if getattr(cfg.model, field, None) is None:
            raise ValueError(f"缺少必要配置: model.{field}")

    # 日志与输出目录
    ckpt_name = os.path.splitext(os.path.basename(cfg.model.ckpt))[0]
    results_root = cfg.model.results_dir if cfg.model.results_dir else "./results"
    eval_dir = os.path.join(results_root, f"eval_{ckpt_name}")
    samples_dir = os.path.join(eval_dir, "samples") if args.save_samples else None
    logger = create_logger(eval_dir)

    logger.info(f"Eval dir: {eval_dir}")
    logger.info(f"Device: {device} | AMP: {amp} | Use EMA: {not args.no_ema}")

    # 固定随机种子
    seed = getattr(cfg.model, "global_seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 构建模型与数据
    model, diffusion, AE = build_model(cfg, device)
    ema_model = copy.deepcopy(model)
    eval_model = load_checkpoint(model, ema_model, cfg.model.ckpt, device, use_ema=not args.no_ema, logger=logger)

    val_bs = cfg.model.batch_size if getattr(cfg.model, "val_batch_size", None) is None else cfg.model.val_batch_size
    if args.val_batch_size is not None:
        val_bs = args.val_batch_size
    val_workers = args.num_workers if args.num_workers is not None else cfg.model.num_workers

    val_dataset = Pre_Intra_Dataset_Ver_128(
        root_dir=cfg.model.data_path,
        files_names_path=cfg.model.val_files_names_path,
        geo_config_path=cfg.model.geo_config_path,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_bs if val_bs is not None else 1,
        num_workers=val_workers,
        shuffle=False,
    )
    logger.info(f"Validation size: {len(val_dataset)}, batch_size={val_bs if val_bs else 1}")

    # 主评估循环
    steps_to_run = args.max_val_batches
    if steps_to_run:
        logger.info(f"只评估前 {steps_to_run} 个 batch 以加速调试")
        val_loader = list(val_loader)[:steps_to_run]

    mean_mae, n_batches = evaluate(
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

    metrics = {"mean_mae": mean_mae, "num_batches": n_batches}
    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation finished. MAE: {mean_mae:.4f}. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()