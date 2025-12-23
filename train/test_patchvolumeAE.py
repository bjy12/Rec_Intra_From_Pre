import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from omegaconf import OmegaConf

# 项目根路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from AutoEncoder.model.PatchVolume import patchvolumeAE
from dataset.vqgan_vertebral_level import VQGAN_Vertebral_Dataset
import pdb

def load_model(ckpt_path, cfg, device):
    """原生 PyTorch 加载 Lightning ckpt 的 state_dict。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model = patchvolumeAE(cfg)
    # 去掉可能的前缀 "model." 等
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {missing}")
    if unexpected:
        print(f"[Warn] Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Test PatchVolumeAE (pure PyTorch eval loop).")
    parser.add_argument("--config", type=str, required=True, help="Config yaml path.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.ckpt).")
    parser.add_argument("--test_list", type=str, required=True, help="Test file list txt.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size for testing.")
    parser.add_argument("--num_workers", type=int, default=0, help="Override num_workers.")
    parser.add_argument("--save_dir", type=str, default=None, help="Optional: save recon samples (nii.gz) here.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    torch.manual_seed(cfg.model.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"CUDA is available! Found {torch.cuda.device_count()} GPU(s)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available! Will use CPU.")

    bs = args.batch_size if args.batch_size is not None else cfg.model.batch_size
    nw = args.num_workers if args.num_workers is not None else cfg.model.num_workers

    # 数据集
    test_dataset = VQGAN_Vertebral_Dataset(
        root_dir=cfg.dataset.root_dir,
        augmentation=False,
        split="val",
        files_names_path=args.test_list,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 模型加载（原生 PyTorch）
    model = load_model(args.ckpt, cfg, device)

    # 精度设置
    precision = None
    if precision == "bf16":
        dtype = torch.bfloat16
        print("Using bfloat16 autocast")
    elif precision == "16" or precision == "fp16":
        dtype = torch.float16
        print("Using float16 autocast")
    else:
        dtype = None
        print("Using float32")

    total_recon_loss = 0.0
    total_perceptual_loss = 0.0
    total_commit = 0.0
    total_perplexity = 0.0
    n_batches = 0

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        try:
            import torchio as tio
        except ImportError:
            tio = None
            print("[Warn] torchio not installed; will not save volumes.")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x = batch["data"].to(device)
            #pdb.set_trace()
            case_name = batch['path'][0].split("\\")[-3]
            level = batch['path'][0].split("\\")[-2]
            type = batch['path'][0].split("\\")[-1].split(".")[0]
            ctx = autocast(device_type=device.type, dtype=dtype) if dtype is not None else torch.no_grad()
            with ctx:
                # forward 返回 recon_loss, x_recon, vq_output, perceptual_loss
                recon_loss, x_recon, vq_output, perceptual_loss = model.forward(x, val=True)

            total_recon_loss += recon_loss.item()
            total_perceptual_loss += perceptual_loss.item()
            total_commit += vq_output["commitment_loss"].item()
            total_perplexity += vq_output["perplexity"].item()
            n_batches += 1
            #pdb.set_trace()

            if save_dir and "path" in batch and dtype is None and "affine" in batch:
                # 仅在有 affine 且使用 float32 时保存（避免精度转换麻烦）
                if "affine" in batch:
                    import numpy as np
                if "affine" in batch and "path" in batch and dtype is None and "affine" in batch:
                    pass
            # 简化：若指定 save_dir，保存第一批的输入与重建
            if save_dir is not None:
                try:
                    import numpy as np
                    import torchio as tio
                    in_aff = batch.get("affine", None)
                    in_path = batch["path"][0] if isinstance(batch["path"], (list, tuple)) else f"sample_{i}.nii.gz"
                    base = os.path.splitext(os.path.basename(in_path))[0]
                    # 保存输入/重建
                    input_img = x.cpu()
                    recon_img = x_recon.cpu()
                    aff = in_aff[0].numpy() if (in_aff is not None) else np.eye(4)
                    save_name = f"{case_name}_{level}_{type}"
                    tio.ScalarImage(tensor=input_img.squeeze(0), affine=aff).save(os.path.join(save_dir, f"{save_name}_input.nii.gz"))
                    tio.ScalarImage(tensor=recon_img.squeeze(0), affine=aff).save(os.path.join(save_dir, f"{save_name}_recon.nii.gz"))
                    print(f"[Info] Saved sample input/recon to {save_dir}")
                except Exception as e:
                    print(f"[Warn] Save failed: {e}")

    if n_batches == 0:
        print("[Warn] No batches processed.")
        return

    print("==== Test Results ====")
    print(f"batches: {n_batches}")
    print(f"avg recon_loss: {total_recon_loss / n_batches:.6f}")
    print(f"avg perceptual_loss: {total_perceptual_loss / n_batches:.6f}")
    print(f"avg commitment_loss: {total_commit / n_batches:.6f}")
    print(f"avg perplexity: {total_perplexity / n_batches:.6f}")


if __name__ == "__main__":
    main()

