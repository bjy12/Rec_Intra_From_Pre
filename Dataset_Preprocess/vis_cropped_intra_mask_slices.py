"""
可视化 crop_intra_ct_and_registred_pre_ct.py 生成的裁剪结果：
- 对每个病例、每个椎体 level，读取裁剪后的 intra/mask（全尺寸版本），取中间切片 (z 方向) 并输出：
  - intra_slice.png       (灰度)
  - mask_slice.png        (标签着色)
  - overlay_slice.png     (mask 叠加在 CT 上)

输入目录（与 crop 脚本保持一致）:
- CROPPED_ROOT/case/level/
    - intra_crop_fullsize.nii.gz
    - mask_crop_fullsize.nii.gz

可选：读取 black_case.txt，跳过黑名单病例。
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Tuple

# ===== 配置 =====
CROPPED_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\cropped_by_mask"
OUT_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\cropped_by_mask_vis"
LEVELS = ("L1", "L2", "L3", "L4", "L5")
INTRA_NAME = "intra_crop_fullsize.nii.gz"
MASK_NAME = "mask_crop_fullsize.nii.gz"
MASKED_INTRA_NAME = "intra_crop_masked_by_dilated.nii.gz"
BLACKLIST_PATH = os.path.join(os.path.dirname(__file__), "black_case.txt")

LABEL_COLORS: Dict[int, Tuple[float, float, float]] = {
    0: (0.0, 0.0, 0.0),
    1: (1.0, 0.0, 0.0),
    2: (0.0, 1.0, 0.0),
    3: (0.0, 0.0, 1.0),
    4: (1.0, 1.0, 0.0),
    5: (1.0, 0.0, 1.0),
    6: (0.0, 1.0, 1.0),
    7: (1.0, 0.5, 0.0),
}
# =================


def load_nifti(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    return arr


def mid_slice(arr: np.ndarray) -> np.ndarray:
    z = arr.shape[0] // 2
    return arr[:,:,z]


def mid_slices_three_axes(arr: np.ndarray):
    """返回轴向/冠状/矢状三个方向的中间切片。"""
    z_mid = arr.shape[0] // 2
    y_mid = arr.shape[1] // 2
    x_mid = arr.shape[2] // 2
    return {
        "axial_z": arr[z_mid, :, :],
        "coronal_y": arr[:, y_mid, :],
        "sagittal_x": arr[:, :, x_mid],
    }


def normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        return (img - vmin) / (vmax - vmin)
    return np.zeros_like(img, dtype=np.float32)


def colorize_mask(mask2d: np.ndarray) -> np.ndarray:
    h, w = mask2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    unique_labels = np.unique(mask2d)
    cmap = cm.get_cmap("tab20")
    for idx, label in enumerate(unique_labels):
        if label in LABEL_COLORS:
            color = LABEL_COLORS[label]
        else:
            color = cmap(idx % 20)[:3]
        rgb[mask2d == label] = color
    return rgb


def save_png(path: str, image: np.ndarray, cmap=None):
    plt.imsave(path, image, cmap=cmap)


def process_one(case: str, blacklist):
    if case in blacklist:
        print(f"[跳过黑名单] case={case}")
        return

    case_dir = os.path.join(CROPPED_ROOT, case)
    if not os.path.isdir(case_dir):
        return
    out_case = os.path.join(OUT_ROOT, case)
    os.makedirs(out_case, exist_ok=True)

    for level in LEVELS:
        level_dir = os.path.join(case_dir, level)
        if not os.path.isdir(level_dir):
            continue
        intra_path = os.path.join(level_dir, INTRA_NAME)
        mask_path = os.path.join(level_dir, MASK_NAME)
        if not (os.path.isfile(intra_path) and os.path.isfile(mask_path)):
            continue

        ct_arr = load_nifti(intra_path)
        mask_arr = load_nifti(mask_path)

        ct_slice = mid_slice(ct_arr)
        mask_slice = mid_slice(mask_arr)
        mask_rgb = colorize_mask(mask_slice)

        # overlay
        ct_norm = ct_slice.astype(np.float32)
        if ct_norm.max() > ct_norm.min():
            ct_norm = (ct_norm - ct_norm.min()) / (ct_norm.max() - ct_norm.min())
        ct_gray_rgb = np.stack([ct_norm]*3, axis=-1)
        overlay = (0.6 * ct_gray_rgb + 0.4 * mask_rgb).clip(0, 1)

        out_level_dir = os.path.join(out_case, level)
        os.makedirs(out_level_dir, exist_ok=True)
        save_png(os.path.join(out_level_dir, "ct_slice.png"), ct_slice, cmap="gray")
        save_png(os.path.join(out_level_dir, "mask_slice.png"), mask_rgb)
        save_png(os.path.join(out_level_dir, "overlay_slice.png"), overlay)

        # 可视化膨胀后 mask 作用下的 intra 体素块（三个方向中间切片）
        masked_path = os.path.join(level_dir, MASKED_INTRA_NAME)
        if os.path.isfile(masked_path):
            masked_arr = load_nifti(masked_path)
            for name, sl in mid_slices_three_axes(masked_arr).items():
                save_png(os.path.join(out_level_dir, f"masked_{name}.png"), normalize01(sl), cmap="gray")


def main():
    blacklist = set()
    if os.path.isfile(BLACKLIST_PATH):
        with open(BLACKLIST_PATH, "r", encoding="utf-8") as f:
            blacklist = {line.strip() for line in f if line.strip()}

    cases = [d for d in os.listdir(CROPPED_ROOT) if os.path.isdir(os.path.join(CROPPED_ROOT, d))]
    cases.sort()
    for case in cases:
        process_one(case, blacklist)
    print("Done. Visualization saved to:", OUT_ROOT)


if __name__ == "__main__":
    main()

