"""
对齐每个病例的椎体级（L1-L5）术中 CT 裁剪块与术前 CT（配准结果），
输出同一空间、同一尺寸的配准对，并保存元数据。
"""

import json
import os
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk

# ========== 配置区域：按需修改 ==========
INTRA_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\cropped_by_mask"
PRE_ROOT = r"D:\Elastic\elastix-5.0.1-win64_exe\elastix-5.0.1-win64\resampled_results_pre_to_intra"
OUT_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\aligned_intra_pre"
LEVELS = ("L1", "L2", "L3", "L4", "L5")
INTRA_FILE = "intra_crop_masked_by_dilated.nii.gz"
INTRA_BBOX = "bbox.json"
PRE_SUBDIR_PATTERN = "{level}_resample_results"
PRE_FILE = "moving_in_fixed.nii.gz"
# 立方体输出尺寸（voxel）
BOX_SIZE = (128, 128, 128)
# 重采样的填充值（非重叠区域）
FILL_VALUE = -1000
# ======================================


def load_image(path: str) -> sitk.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return sitk.ReadImage(path)


def centroid_from_image(img: sitk.Image, threshold: float = 0.0) -> Tuple[int, int, int]:
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    coords = np.argwhere(arr > threshold)
    if coords.size == 0:
        raise ValueError("图像为空，无法计算质心")
    centroid_zyx = coords.mean(axis=0)
    centroid_xyz = centroid_zyx[::-1]  # 转为 x,y,z
    return tuple(np.round(centroid_xyz).astype(int).tolist())


def direction_matrix(direction: Tuple[float, ...]) -> np.ndarray:
    if len(direction) == 9:
        return np.array(direction, dtype=float).reshape(3, 3)
    raise ValueError("方向矩阵维度错误，应为 9 个元素")


def build_reference_from_image(image: sitk.Image, center_index_xyz: Tuple[int, int, int], box_size=BOX_SIZE) -> sitk.Image:
    """以给定图像的 spacing/direction 为基准，构建以质心为中心的参考网格。"""
    spacing = np.array(image.GetSpacing(), dtype=float)
    size = np.array(box_size, dtype=int)
    dir_mat = direction_matrix(image.GetDirection())

    half = (size * spacing) / 2.0
    center_phys = np.array(image.TransformIndexToPhysicalPoint(tuple(int(v) for v in center_index_xyz)))
    origin = center_phys - dir_mat @ half

    ref = sitk.Image(tuple(size.tolist()), image.GetPixelID())
    ref.SetSpacing(tuple(spacing.tolist()))
    ref.SetDirection(image.GetDirection())
    ref.SetOrigin(tuple(origin.tolist()))
    return ref


def resample_to_reference(img: sitk.Image, reference: sitk.Image, fill_value=FILL_VALUE) -> sitk.Image:
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(reference)
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(fill_value)
    return rs.Execute(img)


def fill_neg_to_value(img: sitk.Image, value: float = -500) -> sitk.Image:
    """将图像中小于 0 的体素置为指定值，其余保持不变。"""
    arr = sitk.GetArrayFromImage(img)
    arr = np.where(arr < -1000, value, arr)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def process_one(case: str):
    for level in LEVELS:
        intra_level_dir = os.path.join(INTRA_ROOT, case, level)
        pre_level_dir = os.path.join(PRE_ROOT, case, PRE_SUBDIR_PATTERN.format(level=level))
        intra_path = os.path.join(intra_level_dir, INTRA_FILE)
        bbox_path = os.path.join(intra_level_dir, INTRA_BBOX)
        pre_path = os.path.join(pre_level_dir, PRE_FILE)
        #import pdb
        #pdb.set_trace()
        if not (os.path.isfile(intra_path) and os.path.isfile(pre_path) and os.path.isfile(bbox_path)):
            print(f"[跳过] case={case} level={level}: 缺少文件")
            continue

        intra_img = load_image(intra_path)
        pre_img = load_image(pre_path)
        #import pdb
        #pdb.set_trace()
        try:
            centroid_idx = centroid_from_image(intra_img, threshold=0.0)
        except ValueError as e:
            print(f"[跳过] case={case} level={level}: {e}")
            continue

        ref = build_reference_from_image(intra_img, centroid_idx, box_size=BOX_SIZE)
        #pdb.set_trace()
        # 使用 intra 的 spacing/origin/direction 构建的 bbox（ref）对 pre 进行裁剪/对齐
        intra_aligned = resample_to_reference(intra_img, ref, fill_value=FILL_VALUE)
        pre_aligned = resample_to_reference(pre_img, ref, fill_value=FILL_VALUE)

        # 将 volume 中小于 0 的体素置为 -500
        intra_aligned = fill_neg_to_value(intra_aligned, value=-1000)
        pre_aligned = fill_neg_to_value(pre_aligned, value=-1000)

        out_dir = os.path.join(OUT_ROOT, case, level)
        os.makedirs(out_dir, exist_ok=True)
        sitk.WriteImage(intra_aligned, os.path.join(out_dir, "intra_aligned.nii.gz"))
        sitk.WriteImage(pre_aligned, os.path.join(out_dir, "pre_aligned.nii.gz"))

        with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "case": case,
                    "level": level,
                    "box_size_vox": list(BOX_SIZE),
                    "reference_spacing": list(ref.GetSpacing()),
                    "reference_origin": list(ref.GetOrigin()),
                    "reference_direction": list(ref.GetDirection()),
                    "centroid_index_xyz": list(centroid_idx),
                    "intra_source": intra_path,
                    "pre_source": pre_path,
                    "bbox_source": bbox_path,
                    "fill_value": FILL_VALUE,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    case_names = [d for d in os.listdir(INTRA_ROOT) if os.path.isdir(os.path.join(INTRA_ROOT, d))]
    case_names.sort()
    for case in case_names:
        process_one(case)
    print(f"完成。输出目录: {OUT_ROOT}")


if __name__ == "__main__":
    main()

