import SimpleITK as sitk
import numpy as np
import os
from typing import List, Tuple


# ========== 需要你修改的路径 ==========
# 数据根目录（病例子目录，内含 L1-L5 的 pre/intra）
DATA_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\final_masked_full_volumes"
# elastix 生成的 TransformParameters.0.txt 根目录（对应 run_elastix_batch 的 OUTPUT_ROOT）
TRANSFORM_ROOT = r"D:\Elastic\elastix-5.0.1-win64_exe\elastix-5.0.1-win64\demo_results_batch"
# 重采样结果输出根目录
OUTPUT_ROOT = r"D:\Elastic\elastix-5.0.1-win64_exe\elastix-5.0.1-win64\resampled_results"

# 文件命名模式（与 run_elastix_batch 保持一致）
VERTEBRAE = ("L1", "L2", "L3", "L4", "L5")
FIXED_SUFFIX = "_intra_ct_masked.nii.gz"  # fixed
MOVING_SUFFIX = "_pre_mask.nii.gz"   # moving
# ====================================


def load_elastix_euler3d_transform(param_file):
    """
    从 Elastix TransformParameters.txt 里读取 EulerTransform，
    构造 SimpleITK 的 Euler3DTransform，并返回 transform 与 4x4 仿射矩阵。
    """
    # 读取 parameter map（纯 Python 解析 TransformParameters.0.txt）
    param_map = {}
    with open(param_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("(") and line.endswith(")"):
                items = line[1:-1].strip().split()
                key, values = items[0], items[1:]
                values = [v.strip('"') for v in values]  # 去掉引号
                param_map[key] = values

    # 检查 Transform 类型
    transform_type = param_map["Transform"][0]
    if transform_type != "EulerTransform":
        raise ValueError(f"当前脚本只处理 EulerTransform，检测到: {transform_type}")

    # 读取 transform 参数（6 个）
    # 顺序: [Rx, Ry, Rz, Tx, Ty, Tz]
    params = list(map(float, param_map["TransformParameters"]))
    assert len(params) == 6, f"期望 6 个参数，实际为 {len(params)}"

    rx, ry, rz = params[0], params[1], params[2]
    tx, ty, tz = params[3], params[4], params[5]

    # 旋转中心
    center = list(map(float, param_map["CenterOfRotationPoint"]))
    if len(center) != 3:
        raise ValueError(f"CenterOfRotationPoint 维度错误: {len(center)}")

    # 构造 SimpleITK Euler3DTransform
    euler = sitk.Euler3DTransform()
    euler.SetCenter(center)
    # 注意：SimpleITK 的顺序是 (angleX, angleY, angleZ)
    euler.SetRotation(rx, ry, rz)
    euler.SetTranslation((tx, ty, tz))

    # 从 transform 中获取 3x3 矩阵和 3x1 平移：
    # y = M x + t
    M = np.array(euler.GetMatrix(), dtype=float).reshape(3, 3)
    t = np.array(euler.GetTranslation(), dtype=float).reshape(3)

    # 组装 4x4 仿射矩阵
    affine_4x4 = np.eye(4, dtype=float)
    affine_4x4[:3, :3] = M
    affine_4x4[:3, 3] = t

    return euler, affine_4x4


def build_pairs_from_root(root_dir: str) -> List[Tuple[str, str, str]]:
    """
    扫描 DATA_ROOT 下病例子目录，返回 (case_name, fixed_path, moving_path) 列表。
    """
    pairs: List[Tuple[str, str, str]] = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"DATA_ROOT 不存在: {root_dir}")
    case_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    case_names.sort()

    for case in case_names:
        case_dir = os.path.join(root_dir, case)
        for v in VERTEBRAE:
            fixed = os.path.join(case_dir, f"{v}{FIXED_SUFFIX}")
            moving = os.path.join(case_dir, f"{v}{MOVING_SUFFIX}")
            if os.path.isfile(fixed) and os.path.isfile(moving):
                pairs.append((case, fixed, moving))
    return pairs


def resample_one(fixed_path: str, moving_path: str, transform_path: str, out_dir: str) -> None:
    """应用 transform，把 moving 重采样到 fixed 空间，并保存 affine 矩阵。"""
    if not os.path.isfile(transform_path):
        raise FileNotFoundError(f"未找到 transform: {transform_path}")
    os.makedirs(out_dir, exist_ok=True)

    fixed = sitk.ReadImage(fixed_path)
    moving = sitk.ReadImage(moving_path)

    euler_transform, affine_4x4 = load_elastix_euler3d_transform(transform_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(euler_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    moving_in_fixed = resampler.Execute(moving)

    out_moving = os.path.join(out_dir, "moving_in_fixed.nii.gz")
    sitk.WriteImage(moving_in_fixed, out_moving)
    np.savetxt(os.path.join(out_dir, "affine_matrix_4x4.txt"), affine_4x4, fmt="%.8f")
    print(f"Saved resampled moving -> {out_moving}")

    # 可选：inverse resample fixed -> moving
    try:
        inverse_transform = euler_transform.GetInverse()
        resampler2 = sitk.ResampleImageFilter()
        resampler2.SetReferenceImage(moving)
        resampler2.SetTransform(inverse_transform)
        resampler2.SetInterpolator(sitk.sitkLinear)
        resampler2.SetDefaultPixelValue(0)
        fixed_in_moving = resampler2.Execute(fixed)
        out_fixed = os.path.join(out_dir, "fixed_in_moving.nii.gz")
        sitk.WriteImage(fixed_in_moving, out_fixed)
        print(f"Saved fixed_in_moving -> {out_fixed}")
    except RuntimeError:
        pass


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    pairs = build_pairs_from_root(DATA_ROOT)
    if not pairs:
        raise RuntimeError(f"在 {DATA_ROOT} 下未找到任何 (fixed, moving) 成对文件。")

    for idx, (case, fixed, moving) in enumerate(pairs):
        level = os.path.splitext(os.path.basename(moving))[0].split("_")[0]
        transform_path = os.path.join(TRANSFORM_ROOT, case, f"{level}_registration_results", "TransformParameters.0.txt")
        out_dir = os.path.join(OUTPUT_ROOT, case, f"{level}_resample_results")
        print(f"[{idx+1}/{len(pairs)}] case={case} level={level}")
        resample_one(fixed, moving, transform_path, out_dir)


if __name__ == "__main__":
    main()
