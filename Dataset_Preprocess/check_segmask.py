"""
检查分割结果：遍历病例目录，对每个椎体（L1-L5）的
intra/pre 清洗后掩码进行可视化，生成：
- 合成图（全部标签）
- 单标签图（逐个标签）

输入结构：
ROOT_DIR/
  caseA/
    intra_processed_mask.nii.gz
    pre_processed_mask.nii.gz
  caseB/...

输出：
OUT_DIR/caseA/
  intra_slice.png
  pre_slice.png
"""

import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
# ========== 配置区域：按需修改 ==========
ROOT_DIR = "D:/data_space/Zhongrifriendly/csh/paired_data/CT_mask"
OUT_DIR = "D:/data_space/Zhongrifriendly/csh/paired_data/CT_mask_vis"
LEVELS = ("L1", "L2", "L3", "L4", "L5")
INTRA_SUFFIX = "_intra_mask_cleaned.nii.gz"
PRE_SUFFIX = "_pre_mask_cleaned.nii.gz"

# 自定义标签颜色映射（label_id: (r,g,b) in [0,1]）
LABEL_COLORS = {
    0: (0.0, 0.0, 0.0),       # 背景
    1: (1.0, 0.0, 0.0),       # label 1 -> 红
    2: (0.0, 1.0, 0.0),       # label 2 -> 绿
    3: (0.0, 0.0, 1.0),       # label 3 -> 蓝
    4: (1.0, 1.0, 0.0),       # label 4 -> 黄
    5: (1.0, 0.0, 1.0),       # label 5 -> 品红
    6: (0.0, 1.0, 1.0),       # label 6 -> 青
    7: (1.0, 0.5, 0.0),       # label 7 -> 橙
}
# 若未定义的标签，使用 tab20 colormap 自动分配
# ======================================
# level 叠加时的颜色（按椎体编号），未定义时使用 tab10
LEVEL_COLORS = {lvl: cm.get_cmap("tab10")(i % 10)[:3] for i, lvl in enumerate(LEVELS)}


def load_mask(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到文件: {path}")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    return arr


def get_middle_slice(arr: np.ndarray) -> np.ndarray:
    # 采用 z 方向中间层（arr 格式为 z,y,x）
    z = arr.shape[0] // 2
    return arr[:,:,z]


def combine_masks(mask_list):
    """将同尺寸的多张 3D mask 合并成一张（逐体素取最大值）。"""
    if not mask_list:
        return None
    base_shape = mask_list[0].shape
    combined = np.zeros(base_shape, dtype=mask_list[0].dtype)
    for m in mask_list:
        if m.shape != base_shape:
            raise ValueError(f"mask shape 不一致: {m.shape} vs {base_shape}")
        combined = np.maximum(combined, m)
    return combined


def save_level_overlay(level_slices, out_path):
    """
    将各 level 的中间切片叠加到同一张彩色图像，按 level 着色。
    level_slices: List[Tuple[level_name, slice2d]]
    """
    if not level_slices:
        return
    h, w = level_slices[0][1].shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for level, slice2d in level_slices:
        mask = slice2d > 0
        if not mask.any():
            continue
        color = LEVEL_COLORS.get(level, cm.get_cmap("tab10")(hash(level) % 10)[:3])
        rgb[mask] = color
    plt.imsave(out_path, rgb)


def colorize_mask(mask2d: np.ndarray) -> np.ndarray:
    """返回 (H,W,3) RGB in [0,1]"""
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


def save_slice_png(mask_arr: np.ndarray, out_path: str):
    mask2d = get_middle_slice(mask_arr)
    rgb = colorize_mask(mask2d)
    plt.imsave(out_path, rgb)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    case_names = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    case_names.sort()

    for case in tqdm(case_names, desc="Cases"):
        case_dir = os.path.join(ROOT_DIR, case)
        out_dir = os.path.join(OUT_DIR, case)
        os.makedirs(out_dir, exist_ok=True)

        intra_list = []
        pre_list = []
        intra_level_slices = []
        pre_level_slices = []

        for level in LEVELS:
            intra_path = os.path.join(case_dir, f"{level}{INTRA_SUFFIX}")
            pre_path = os.path.join(case_dir, f"{level}{PRE_SUFFIX}")

            for name, path in (("intra", intra_path), ("pre", pre_path)):
                try:
                    arr = load_mask(path)
                    if name == "intra":
                        intra_list.append(arr)
                        intra_level_slices.append((level, get_middle_slice(arr)))
                    else:
                        pre_list.append(arr)
                        pre_level_slices.append((level, get_middle_slice(arr)))
                    # 全标签合成可视化
                    out_png = os.path.join(out_dir, f"{level}_{name}_all_labels.png")
                    save_slice_png(arr, out_png)

                    # 针对每个标签单独可视化
                    labels = np.unique(arr)
                    slice2d = get_middle_slice(arr)
                    for lbl in labels:
                        if lbl == 0:
                            continue  # 默认跳过背景
                        mask = (slice2d == lbl).astype(np.float32)
                        if mask.max() == 0:
                            continue
                        rgb = np.stack([mask, mask, mask], axis=-1)
                        out_lbl = os.path.join(out_dir, f"{level}_{name}_label{int(lbl)}.png")
                        plt.imsave(out_lbl, rgb)
                except Exception as e:
                    print(f"[跳过] case={case} level={level} {name}: {e}")

        # 合并所有 level 的 mask，输出整体可视化
        for name, lst in (("intra", intra_list), ("pre", pre_list)):
            try:
                combined = combine_masks(lst)
                if combined is None:
                    continue
                out_png = os.path.join(out_dir, f"{name}_all_levels_combined.png")
                save_slice_png(combined, out_png)
            except Exception as e:
                print(f"[跳过合成] case={case} {name}: {e}")

        # 按 level 着色叠加的整体可视化
        for name, level_slices in (("intra", intra_level_slices), ("pre", pre_level_slices)):
            try:
                if not level_slices:
                    continue
                out_png = os.path.join(out_dir, f"{name}_all_levels_overlay.png")
                save_level_overlay(level_slices, out_png)
            except Exception as e:
                print(f"[跳过level叠加] case={case} {name}: {e}")


if __name__ == "__main__":
    main()