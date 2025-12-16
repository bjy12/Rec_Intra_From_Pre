"""
批量根据对齐后的 mask（moving_in_fixed）在 intra CT 上裁剪对应椎体的 ROI，
输出裁剪后的体数据与 bbox 信息。

输入目录：
- INTRA_ROOT: D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\processed_176_1_volume
    - caseA/intra_processed_volume.nii.gz
    - caseB/...
- MASK_ROOT: D:\Elastic\elastix-5.0.1-win64_exe\elastix-5.0.1-win64\resampled_results
    - caseA/L1_resample_results/moving_in_fixed.nii.gz  (mask, 已与 intra 对齐)
    - caseA/L2_resample_results/moving_in_fixed.nii.gz
    - ...

输出：
- OUT_ROOT/case/level/
    - intra_crop.nii.gz        (裁剪后的 intra CT)
    - mask_crop.nii.gz         (裁剪后的 mask，若需要)
    - bbox.json                (物理坐标与体素坐标 bbox 记录)
"""

import os
import json
import csv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from typing import Tuple

# ========== 配置区域：按需修改 ==========
INTRA_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\processed_176_1_volume"
MASK_ROOT = r"D:\Elastic\elastix-5.0.1-win64_exe\elastix-5.0.1-win64\resampled_results"
OUT_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\cropped_by_mask"
LEVEL_NAMES = ("L1", "L2", "L3", "L4", "L5")
INTRA_FILENAME = "intra_processed_volume.nii.gz"
MASK_FILENAME = "moving_in_fixed.nii.gz"
# 裁剪时在各方向的 margin（voxel），可根据需要调整
MARGIN_VOX = (4, 4, 4)
# mask 膨胀半径（voxel），用于计算膨胀后的体积与提取重叠体素块
DILATE_RADIUS_VOX = (2, 2, 2)
# 统计文件输出路径
STATS_CSV = os.path.join(OUT_ROOT, "bbox_stats.csv")
# 黑名单（出现在此列表的 case 将被跳过），默认读取同目录 black_case.txt
BLACKLIST_PATH = os.path.join(os.path.dirname(__file__), "black_case.txt")
# ======================================


def load_nifti(path: str) -> sitk.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到文件: {path}")
    return sitk.ReadImage(path)


def mask_bbox(mask_img: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:
    """返回 mask 的体素 bbox (min_xyz, max_xyz) 闭区间索引。"""
    arr = sitk.GetArrayFromImage(mask_img)  # z, y, x
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        raise ValueError("mask 中没有非零体素")
    zyx_min = coords.min(axis=0)
    zyx_max = coords.max(axis=0)
    # 转换为 x,y,z 顺序
    xyz_min = zyx_min[::-1]
    xyz_max = zyx_max[::-1]
    return xyz_min, xyz_max


def expand_bbox(xyz_min, xyz_max, margin, size):
    """在给定 margin 下扩展 bbox，限制在体积范围内。"""
    xyz_min = np.maximum(xyz_min - np.array(margin), 0)
    xyz_max = np.minimum(xyz_max + np.array(margin), np.array(size) - 1)
    return xyz_min.astype(int), xyz_max.astype(int)


def crop_with_mask(intra_img: sitk.Image, mask_img: sitk.Image, margin=MARGIN_VOX, dilation_radius=None):
    """可选先对 mask 膨胀，再按 bbox+margin 裁剪，并返回重叠体素块。"""
    radius = _normalize_radius(dilation_radius) if dilation_radius is not None else (0, 0, 0)
    use_dilate = any(r > 0 for r in radius)
    mask_for_bbox = sitk.BinaryDilate(mask_img > 0, radius) if use_dilate else mask_img

    xyz_min, xyz_max = mask_bbox(mask_for_bbox)
    size = np.array(intra_img.GetSize())  # x,y,z
    xyz_min, xyz_max = expand_bbox(xyz_min, xyz_max, margin, size)

    region_size = (xyz_max - xyz_min + 1).tolist()
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(region_size)
    extractor.SetIndex(xyz_min.tolist())

    intra_crop = extractor.Execute(intra_img)
    mask_crop = extractor.Execute(mask_img)
    mask_crop_dilated = extractor.Execute(mask_for_bbox) if use_dilate else mask_crop

    # 生成与原始尺寸一致的全尺寸裁剪图，保持空间信息不变（位置不变）
    intra_full = sitk.Image(intra_img.GetSize(), intra_img.GetPixelID())
    intra_full.CopyInformation(intra_img)
    mask_full = sitk.Image(mask_img.GetSize(), mask_img.GetPixelID())
    mask_full.CopyInformation(mask_img)

    intra_full = sitk.Paste(intra_full, intra_crop, intra_crop.GetSize(), destinationIndex=xyz_min.tolist())
    mask_full = sitk.Paste(mask_full, mask_crop, mask_crop.GetSize(), destinationIndex=xyz_min.tolist())

    # 记录物理 bbox（以 intra 原始空间）
    origin = np.array(intra_img.TransformIndexToPhysicalPoint(xyz_min.tolist()))
    spacing = np.array(intra_img.GetSpacing())
    direction = intra_img.GetDirection()

    bbox_info = {
        "xyz_min_index": xyz_min.tolist(),
        "xyz_max_index": xyz_max.tolist(),
        "origin_at_min": origin.tolist(),
        "spacing": spacing.tolist(),
        "direction": direction,
        "size": region_size,
    }
    # 基于膨胀后 mask 获取与 intra 重叠的体素块
    intra_masked_crop = sitk.Mask(intra_crop, mask_crop_dilated > 0 , outsideValue=-500)

    return intra_crop, mask_crop, intra_full, mask_full, bbox_info, mask_crop_dilated, intra_masked_crop


def _normalize_radius(radius):
    """将膨胀半径标准化为 (x, y, z) 元组。"""
    if radius is None:
        return (0, 0, 0)
    if isinstance(radius, int):
        return (radius, radius, radius)
    radius = tuple(int(r) for r in radius)
    if len(radius) == 1:
        return (radius[0],) * 3
    if len(radius) != 3:
        raise ValueError("dilation radius 必须为 int、长度为1或长度为3的序列")
    return radius


def mask_volume_stats(mask_img: sitk.Image, dilation_radius=DILATE_RADIUS_VOX):
    """
    计算 mask 的体素数量与体积，并可选对 mask 进行膨胀后再计算。
    返回 dict，包含原始与膨胀后的体素数与体积（mm^3）。
    """
    radius = _normalize_radius(dilation_radius)
    mask_bin = mask_img > 0
    spacing = np.array(mask_img.GetSpacing())
    voxel_volume_mm3 = float(np.prod(spacing))

    voxels = int(sitk.GetArrayViewFromImage(mask_bin).sum())
    volume_mm3 = voxels * voxel_volume_mm3

    if any(r > 0 for r in radius):
        mask_dilated = sitk.BinaryDilate(mask_bin, radius)
        voxels_dilated = int(sitk.GetArrayViewFromImage(mask_dilated).sum())
        volume_dilated_mm3 = voxels_dilated * voxel_volume_mm3
    else:
        voxels_dilated = voxels
        volume_dilated_mm3 = volume_mm3

    return {
        "voxels": voxels,
        "volume_mm3": volume_mm3,
        "voxels_dilated": voxels_dilated,
        "volume_dilated_mm3": volume_dilated_mm3,
        "dilation_radius_vox": radius,
    }


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        return (arr - vmin) / (vmax - vmin)
    return np.zeros_like(arr, dtype=np.float32)


def save_middle_slices(masked_img: sitk.Image, out_dir: str, prefix="intra_masked"):
    """
    保存膨胀后 mask 作用下 intra 的三个方向中间切片可视化。
    输出：{prefix}_axial_z.png / _coronal_y.png / _sagittal_x.png
    """
    arr = sitk.GetArrayFromImage(masked_img)  # z,y,x
    z_mid = arr.shape[0] // 2
    y_mid = arr.shape[1] // 2
    x_mid = arr.shape[2] // 2
    slices = {
        "axial_z": arr[z_mid, :, :],
        "coronal_y": arr[:, y_mid, :],
        "sagittal_x": arr[:, :, x_mid],
    }
    os.makedirs(out_dir, exist_ok=True)
    for name, sl in slices.items():
        plt.imsave(os.path.join(out_dir, f"{prefix}_{name}.png"), normalize01(sl), cmap="gray")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    case_names = [d for d in os.listdir(MASK_ROOT) if os.path.isdir(os.path.join(MASK_ROOT, d))]
    case_names.sort()

    # 读取黑名单
    blacklist = set()
    if os.path.isfile(BLACKLIST_PATH):
        with open(BLACKLIST_PATH, "r", encoding="utf-8") as f:
            blacklist = {line.strip() for line in f if line.strip()}

    stats_rows = []

    for case in case_names:
        if case in blacklist:
            print(f"[跳过黑名单] case={case}")
            continue
        intra_path = os.path.join(INTRA_ROOT, case, INTRA_FILENAME)
        if not os.path.isfile(intra_path):
            print(f"[跳过] 未找到 intra: {intra_path}")
            continue
        intra_img = load_nifti(intra_path)

        for level in LEVEL_NAMES:
            mask_dir = os.path.join(MASK_ROOT, case, f"{level}_resample_results")
            mask_path = os.path.join(mask_dir, MASK_FILENAME)
            if not os.path.isfile(mask_path):
                print(f"[跳过] 未找到 mask: {mask_path}")
                continue

            mask_img = load_nifti(mask_path)
            try:
                intra_crop, mask_crop, intra_full, mask_full, bbox_info, mask_crop_dilated, intra_masked_crop = crop_with_mask(
                    intra_img, mask_img, margin=MARGIN_VOX, dilation_radius=DILATE_RADIUS_VOX
                )
            except ValueError as e:
                print(f"[跳过] case={case} level={level}: {e}")
                continue

            volume_info = mask_volume_stats(mask_img, dilation_radius=DILATE_RADIUS_VOX)
            bbox_info.update({
                "mask_voxels": volume_info["voxels"],
                "mask_volume_mm3": volume_info["volume_mm3"],
                "mask_dilated_voxels": volume_info["voxels_dilated"],
                "mask_dilated_volume_mm3": volume_info["volume_dilated_mm3"],
                "mask_dilation_radius_vox": volume_info["dilation_radius_vox"],
            })

            out_dir = os.path.join(OUT_ROOT, case, level)
            os.makedirs(out_dir, exist_ok=True)
            sitk.WriteImage(intra_crop, os.path.join(out_dir, "intra_crop.nii.gz"))
            sitk.WriteImage(mask_crop, os.path.join(out_dir, "mask_crop.nii.gz"))
            sitk.WriteImage(mask_crop_dilated, os.path.join(out_dir, "mask_crop_dilated.nii.gz"))
            sitk.WriteImage(intra_masked_crop, os.path.join(out_dir, "intra_crop_masked_by_dilated.nii.gz"))
            # 在写全尺寸文件前，保存 intra_masked_crop 的三个方向中间切片可视化
            save_middle_slices(intra_masked_crop, out_dir, prefix="intra_masked")
            # 全尺寸版本（与原始 intra 尺寸/坐标一致，未移动）
            sitk.WriteImage(intra_full, os.path.join(out_dir, "intra_crop_fullsize.nii.gz"))
            sitk.WriteImage(mask_full, os.path.join(out_dir, "mask_crop_fullsize.nii.gz"))
            with open(os.path.join(out_dir, "bbox.json"), "w", encoding="utf-8") as f:
                json.dump(bbox_info, f, indent=2)

            size_vox = np.array(bbox_info["size"])
            spacing = np.array(bbox_info["spacing"])
            size_mm = size_vox * spacing
            stats_rows.append({
                "case": case,
                "level": level,
                "size_x_vox": int(size_vox[0]),
                "size_y_vox": int(size_vox[1]),
                "size_z_vox": int(size_vox[2]),
                "size_x_mm": float(size_mm[0]),
                "size_y_mm": float(size_mm[1]),
                "size_z_mm": float(size_mm[2]),
                "voxel_volume": int(np.prod(size_vox)),
                "mask_voxels": int(volume_info["voxels"]),
                "mask_volume_mm3": float(volume_info["volume_mm3"]),
                "mask_dilated_voxels": int(volume_info["voxels_dilated"]),
                "mask_dilated_volume_mm3": float(volume_info["volume_dilated_mm3"]),
            })

            print(f"[保存] case={case} level={level} -> {out_dir}")

    if stats_rows:
        os.makedirs(os.path.dirname(STATS_CSV), exist_ok=True)
        with open(STATS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stats_rows)
        print(f"[统计] 已保存 bbox 尺寸汇总: {STATS_CSV}")


if __name__ == "__main__":
    main()
