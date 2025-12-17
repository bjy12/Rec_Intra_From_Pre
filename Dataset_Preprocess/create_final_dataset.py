"""
创建最终的vertebral level 训练数据集:
case_name: L1-L5
Lx: 
包含 术前ct(Lx_pre_ct_maked)
, 对齐到术中空间的ct(pre_aligned)
, 对齐到术中空间的mask(cropped_mask) , 
, 术中ct (intra_aligned) ,
, bbox drr(Level_x_roi.pkl)
"""

import os
import shutil
import SimpleITK as sitk
import numpy as np
import json
import pickle
import yaml
import tqdm

# ========== 配置区域：按需修改 ==========
#*  get  aligned intra space   pre ct and  intra ct 
ALIGNED_PRE_INTRA_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\aligned_intra_pre"
#*  get  aligned intra space   pre mask 
ALIGNED_PRE_ALIGNED_PRE_MASK_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\cropped_by_mask"
#*  pre ct 
PRE_CT_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\final_masked_full_volumes"
#*  get  drr roi 
DRR_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\label_drr_roi_600"
OUT_ROOT = r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\final_dataset"
LEVELS = ("L1", "L2", "L3", "L4", "L5")
# ======================================

def create_final_dataset():
    case_names = os.listdir(ALIGNED_PRE_INTRA_ROOT)
    case_names = [c for c in case_names if os.path.isdir(os.path.join(ALIGNED_PRE_INTRA_ROOT, c))]
    print(f"Found {len(case_names)} cases.")
    for case in case_names:
        for level in LEVELS:
            #* 
            aligned_pre_path = os.path.join(ALIGNED_PRE_INTRA_ROOT, case, level, "intra_aligned.nii.gz")
            aligned_intra_path = os.path.join(ALIGNED_PRE_INTRA_ROOT, case, level, "pre_aligned.nii.gz")
            #* pre mask
            aligned_pre_mask_path = os.path.join(ALIGNED_PRE_ALIGNED_PRE_MASK_ROOT, case, level, "mask_crop.nii.gz")
            #! todo     
            pre_ct = os.path.join(PRE_CT_ROOT, case, f"{level}_pre_ct_masked.nii.gz")
            #* drr 
            drr_path = os.path.join(DRR_ROOT, case, f"Level_{level}_roi.pkl")
            #* out path
            out_path = os.path.join(OUT_ROOT, case, level)
            os.makedirs(out_path, exist_ok=True)

            files_to_copy = {
                "aligned_pre": aligned_pre_path,
                "aligned_intra": aligned_intra_path,
                "aligned_pre_mask": aligned_pre_mask_path,
                "pre_ct_masked": pre_ct,
                "drr_bbox": drr_path,
            }

            for tag, src in files_to_copy.items():
                if not os.path.exists(src):
                    print(f"[Warn] Missing {tag} for {case}-{level}: {src}")
                    continue
                dst = os.path.join(out_path, os.path.basename(src))
                try:
                    shutil.copy2(src, dst)
                    print(f"[Copy] {case}-{level} {tag} -> {dst}")
                except Exception as e:
                    print(f"[Error] Copy {tag} for {case}-{level} failed: {e}")
# ===============================
if __name__ == "__main__":
    create_final_dataset()