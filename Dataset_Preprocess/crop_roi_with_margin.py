import SimpleITK as sitk
import os
import numpy as np
from tqdm import tqdm

def dilate_mask(mask_img, dilation_mm):
    """
    对 Mask 进行二值膨胀，定义感兴趣区域 (ROI)
    """
    spacing = mask_img.GetSpacing()
    kernel_radius = [int(np.ceil(dilation_mm / s)) for s in spacing]
    
    dilater = sitk.BinaryDilateImageFilter()
    dilater.SetKernelType(sitk.sitkBall)
    dilater.SetKernelRadius(kernel_radius)
    dilater.SetForegroundValue(1)
    
    dilated_mask = dilater.Execute(mask_img)
    return dilated_mask

def mask_ct_volume(ct_img, mask_img, background_value=-1000):
    """
    使用 Mask 对 CT 进行掩膜操作 (保留 Mask 内部，外部置为背景值)
    返回的图像尺寸与原图一致。
    """
    masker = sitk.MaskImageFilter()
    masker.SetMaskingValue(0) # Mask=0 (背景) 对应区域会被替换
    masker.SetOutsideValue(background_value) # 替换为 -1000
    
    masked_ct = masker.Execute(ct_img, mask_img)
    return masked_ct

def get_bbox_size_from_mask(mask_img):
    """
    从 Mask 计算有效包围盒尺寸 (用于统计)
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_img)
    
    if not stats.HasLabel(1):
        return None
        
    # 获取 BBox: (x, y, z, w, h, d)
    bbox = stats.GetBoundingBox(1)
    # 返回 Size: (SizeX, SizeY, SizeZ)
    roi_size = [bbox[3], bbox[4], bbox[5]]
    
    return roi_size

def process_single_vertebra(
    ct_img, 
    raw_mask_img, 
    patient_out_dir, 
    label_name, 
    margin_mm, 
    mask_background,
    mode
):
    """
    处理单个椎骨的核心流程 (全尺寸保存模式)
    """
    # 1. 膨胀 Mask (定义保留区域)
    dilated_mask = dilate_mask(raw_mask_img, margin_mm)
    
    # 2. 对 CT 进行 Masking (抠图)
    #    保留 Mask 内像素，Mask 外置为 -1000，保持原图 Shape 和 Spacing
    if mask_background:
        processed_ct = mask_ct_volume(ct_img, dilated_mask, background_value=-1000)
    else:
        processed_ct = ct_img

    # 3. [统计] 计算有效区域大小
    #    虽然保存的是全尺寸图，但我们统计的是“包含信息的区域”有多大
    roi_size = get_bbox_size_from_mask(dilated_mask)
    
    if roi_size is None:
        return False, None # 空 Mask

    # 4. [修改] 直接保存全尺寸图像 (不进行 Crop)
    #    文件名保持不变，内容变为全尺寸 Masked CT
    ct_save_path = os.path.join(patient_out_dir, f"{label_name}_{mode}_ct_masked.nii.gz")
    mask_save_path = os.path.join(patient_out_dir, f"{label_name}_{mode}_mask.nii.gz")
    
    sitk.WriteImage(processed_ct, ct_save_path)
    sitk.WriteImage(raw_mask_img, mask_save_path) # 保存原始分割 Mask
    
    return True, roi_size

def run_pipeline(
    ct_root_dir,
    mask_root_dir,
    output_root_dir,
    report_file_path,
    labels_list=['L1', 'L2', 'L3', 'L4', 'L5'],
    margin_mm=10.0,
    ct_filename="image.nii.gz",
    mask_suffix="_mask_cleaned.nii",
    mask_background=True,
    mode='pre',
    blacklist_path=None
):
    patient_list = os.listdir(mask_root_dir)
    # 加载黑名单
    blacklist = set()
    if blacklist_path and os.path.exists(blacklist_path):
        with open(blacklist_path, 'r', encoding='utf-8') as f:
            blacklist = {line.strip() for line in f if line.strip()}

    print(f"🚀 开始处理: Masking (全尺寸) + Stats...")
    print(f"📂 CT 源: {ct_root_dir}")
    print(f"📝 统计报告: {report_file_path}")
    
    # 用于记录有效 ROI 尺寸
    all_sizes = [] 

    with open(report_file_path, 'w', encoding='utf-8') as f_report:
        # 写入表头
        f_report.write(f"{'PatientID':<20} | {'Vert':<6} | {'ROI X':<8} | {'ROI Y':<8} | {'ROI Z':<8}\n")
        f_report.write("-" * 60 + "\n")
        
        for patient_id in tqdm(patient_list):
            if patient_id in blacklist:
                # 跳过黑名单 case
                continue
            ct_path = os.path.join(ct_root_dir, patient_id, ct_filename)
            patient_mask_dir = os.path.join(mask_root_dir, patient_id)
            
            if not os.path.exists(ct_path):
                continue
                
            try:
                ct_img = sitk.ReadImage(ct_path)
                patient_out_dir = os.path.join(output_root_dir, patient_id)
                os.makedirs(patient_out_dir, exist_ok=True)
                
                for label_name in labels_list:
                    mask_name = f"{label_name}{mask_suffix}"
                    mask_path = os.path.join(patient_mask_dir, mask_name)
                    
                    if not os.path.exists(mask_path):
                        continue
                    
                    raw_mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
                    
                    success, roi_size = process_single_vertebra(
                        ct_img, raw_mask, patient_out_dir, 
                        label_name, margin_mm, mask_background,mode
                    )
                    
                    if success:
                        # 写入报告 (记录有效 ROI 的尺寸)
                        size_x, size_y, size_z = roi_size
                        f_report.write(f"{patient_id:<20} | {label_name:<6} | {size_x:<8} | {size_y:<8} | {size_z:<8}\n")
                        all_sizes.append(roi_size)
                    
            except Exception as e:
                print(f"❌ Error {patient_id}: {e}")

    # --- 打印全局统计 ---
    if len(all_sizes) > 0:
        all_sizes_arr = np.array(all_sizes)
        max_size = np.max(all_sizes_arr, axis=0) # [MaxX, MaxY, MaxZ]
        
        print("\n" + "="*50)
        print("📊 全局 ROI 尺寸统计 (Global Stats)")
        print("="*50)
        print(f"说明: 输出文件为全尺寸，以下统计为'有效内容区域'的大小")
        print(f"处理总数: {len(all_sizes)}")
        print(f"最大有效 ROI (Max Size): X={max_size[0]}, Y={max_size[1]}, Z={max_size[2]}")
        print("="*50)

# ================= 配置区 =================
if __name__ == "__main__":
    # 1. 原始 CT 路径 (保持 Spacing 和 Shape 的基准)
    CT_ROOT = "D:/data_space/Zhongrifriendly/paired_data_cropped_176_1/processed_176_1_volume"
    
    # 2. 分节 Mask 路径
    MASK_ROOT = "D:/data_space/Zhongrifriendly/paired_data_cropped_176_1/separate_masks_cleaned"
    
    # 3. 输出路径
    OUTPUT_ROOT = "D:/data_space/Zhongrifriendly/paired_data_cropped_176_1/final_masked_full_volumes"
    
    # 4. 统计报告保存路径
    REPORT_PATH = "D:/data_space/Zhongrifriendly/paired_data_cropped_176_1/roi_size_stats.txt"

    # 5. 黑名单路径（命中则跳过）
    BLACKLIST_PATH = "./Dataset_Preprocess/black_case.txt"
    
    # 6. 核心参数
    MARGIN_MM = 2.5       # 膨胀半径 (保留多少周围组织)
    MASK_BACKGROUND = True # 必须为 True 才能实现背景置为 -1000
    
    mode = 'intra'

    CT_FILENAME = f"{mode}_processed_volume.nii.gz"
    MASK_SUFFIX = f"_{mode}_mask_cleaned.nii.gz" # 对应 separate_masks_cleaned 里的文件名
    
    run_pipeline(
        CT_ROOT, MASK_ROOT, OUTPUT_ROOT, REPORT_PATH,
        margin_mm=MARGIN_MM,
        mask_background=MASK_BACKGROUND,
        ct_filename=CT_FILENAME, 
        mask_suffix=MASK_SUFFIX,
        mode=mode,
        blacklist_path=BLACKLIST_PATH
    )