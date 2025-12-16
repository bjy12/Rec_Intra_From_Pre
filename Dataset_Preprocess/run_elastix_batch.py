"""
批量调用 elastix 对成对的 CT 体进行刚性配准，并读取 TransformParameters.0.txt
中的欧拉角/平移/旋转中心参数。

使用方法：
1) 修改下面的路径：
   - ELASTIX_EXE：elastix.exe 的完整路径
   - PARAM_FILE：参数文件，如 reg_config/parameters_Rigid.txt
   - DATA_ROOT：病例根目录，结构类似：
       DATA_ROOT/
         caseA/
           L1_intra_ct_masked.nii.gz
           L1_pre_ct_masked.nii.gz
           ...
           L5_intra_ct_masked.nii.gz
           L5_pre_ct_masked.nii.gz
         caseB/...
   - OUTPUT_ROOT：输出目录，每对数据会在子目录下生成 TransformParameters.0.txt
   - VERTEBRAE / FIXED_SUFFIX / MOVING_SUFFIX：根据命名规则调整
2) 运行：
   python run_elastix_batch.py
3) 结果：
   - 每对数据的 TransformParameters.0.txt 保存在对应子目录
   - 在 OUTPUT_ROOT 下保存 summary.csv，包含 Rx,Ry,Rz,Tx,Ty,Tz 和旋转中心
"""

import os
import csv
import subprocess
from typing import List, Tuple, Dict, Any


# ========== 需要你修改的路径 ==========
ELASTIX_EXE = r"D:\Elastic\elastix-5.0.1-win64_exe\elastix-5.0.1-win64\elastix.exe"
PARAM_FILE = r"D:\Elastic\elastix-5.0.1-win64_exe\elastix-5.0.1-win64\reg_config\parameters_Rigid.txt"

# 病例根目录，子目录为各 case（如 anlinv），子目录下包含 L1-L5 的 pre/intra 文件
DATA_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\final_masked_full_volumes"

# 文件命名模式（可按需修改）
VERTEBRAE = ("L1", "L2", "L3", "L4", "L5")
FIXED_SUFFIX = "_intra_ct_masked.nii.gz"  # 作为 fixed
MOVING_SUFFIX = "_pre_ct_masked.nii.gz"   # 作为 moving

OUTPUT_ROOT = r"D:\Elastic\elastix-5.0.1-win64_exe\elastix-5.0.1-win64\demo_results_batch"
# =====================================


def parse_elastix_param_file(path: str) -> Dict[str, Any]:
    """纯 Python 解析 TransformParameters.0.txt，返回键值表。"""
    param_map: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("(") and line.endswith(")"):
                items = line[1:-1].strip().split()
                key, values = items[0], items[1:]
                values = [v.strip('"') for v in values]
                param_map[key] = values
    return param_map


def run_elastix_one(fixed: str, moving: str, out_dir: str) -> Dict[str, Any]:
    """对单对 fixed/moving 调用 elastix，并解析 TransformParameters.0.txt。"""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        ELASTIX_EXE,
        "-f", fixed,
        "-m", moving,
        "-out", out_dir,
        "-p", PARAM_FILE,
    ]
    subprocess.run(cmd, check=True)

    tp_path = os.path.join(out_dir, "TransformParameters.0.txt")
    if not os.path.exists(tp_path):
        raise FileNotFoundError(f"未找到 {tp_path}")

    param_map = parse_elastix_param_file(tp_path)
    tparams = list(map(float, param_map["TransformParameters"]))  # [Rx,Ry,Rz,Tx,Ty,Tz]
    center = list(map(float, param_map["CenterOfRotationPoint"]))

    return {
        "fixed": fixed,
        "moving": moving,
        "out_dir": out_dir,
        "Rx": tparams[0],
        "Ry": tparams[1],
        "Rz": tparams[2],
        "Tx": tparams[3],
        "Ty": tparams[4],
        "Tz": tparams[5],
        "CenterX": center[0],
        "CenterY": center[1],
        "CenterZ": center[2],
    }

def build_pairs_from_root(root_dir: str) -> List[Tuple[str, str]]:
    """扫描根目录下的病例子目录，按命名模式生成 (fixed, moving) 列表。"""
    pairs: List[Tuple[str, str]] = []
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
                pairs.append((fixed, moving))
            else:
                # 缺少任一文件则跳过该 vertebra
                continue
    return pairs


def save_summary(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    results: List[Dict[str, Any]] = []

    pairs = build_pairs_from_root(DATA_ROOT)
    if not pairs:
        raise RuntimeError(f"在 {DATA_ROOT} 下未找到任何 (fixed, moving) 成对文件，请检查命名或路径。")
    #import pdb
    #pdb.set_trace()
    for idx, (fixed, moving) in enumerate(pairs):
        #pdb.set_trace()
        case_name = moving.split("\\")[-2]

        level = os.path.splitext(os.path.basename(moving))[0].split("_")[0]
        out_dir = os.path.join(OUTPUT_ROOT, case_name, f"{level}_registration_results")
        print(f"[{idx+1}/{len(pairs)}] running elastix -> {out_dir}")
        res = run_elastix_one(fixed, moving, out_dir)
        results.append(res)

    summary_path = os.path.join(OUTPUT_ROOT, "summary.csv")
    save_summary(results, summary_path)
    print(f"完成，结果汇总保存在: {summary_path}")


if __name__ == "__main__":
    main()

