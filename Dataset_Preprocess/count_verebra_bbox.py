from sys import prefix
import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm

def clean_and_measure_vertebra(mask_img, label_id):
    """
    对特定 Label 执行：二值化 -> 最大连通域提取 -> 尺寸测量
    返回: 
        - cleaned_mask (SimpleITK.Image): 清洗后的二值 Mask (如果不存在则为None)
        - dims_mm (list): [Depth, Height, Width] 物理尺寸 (mm)
    """
    # 1. 二值化 (0/1)
    binary_mask = (mask_img == label_id)
    
    # 2. 计算连通域
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_mask = cc_filter.Execute(binary_mask)
    
    if cc_filter.GetObjectCount() == 0:
        return None, None # 该标签不存在
        
    # 3. 按面积排序 (Label 1 是最大的)
    relabel_filter = sitk.RelabelComponentImageFilter()
    labeled_mask = relabel_filter.Execute(labeled_mask)
    
    # 4. 提取最大连通域 (Label 1)
    # 这一步去除了小的噪点和碎片
    largest_component = (labeled_mask == 1)
    
    # 5. 计算 Bounding Box 尺寸
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(largest_component)
    
    if not stats.HasLabel(1):
        return None, None
        
    # SITK BBox: (x, y, z, w, h, d)
    bbox = stats.GetBoundingBox(1)
    w_px, h_px, d_px = bbox[3], bbox[4], bbox[5]
    
    # 6. 转换为物理尺寸
    spacing = mask_img.GetSpacing() # (sx, sy, sz)
    w_mm = w_px * spacing[0]
    h_mm = h_px * spacing[1]
    d_mm = d_px * spacing[2]
    
    # 确保返回的是二值图像 (uint8)
    cleaned_mask = sitk.Cast(largest_component, sitk.sitkUInt8)
    
    return cleaned_mask, [d_mm, h_mm, w_mm]

def run_dataset_processing(
    image_root_dir, 
    mask_root_dir, 
    output_root_dir,
    report_file_path,
    labels_dict,
    save_masks=False,  # [开关] 是否保存清洗后的 Mask
    prefix_mode='pre'
):
    # 1. 准备工作
    file_list = os.listdir(image_root_dir)
    print(f"🚀 开始处理 {len(file_list)} 个样本...")
    print(f"📂 Mask 输入: {mask_root_dir}")
    if save_masks:
        print(f"💾 Mask 保存: {output_root_dir}")
    print(f"📝 报告输出: {report_file_path}")
    
    # 统计容器
    all_dims_mm = []
    missing_files = []
    
    # 打开报告文件准备写入
    with open(report_file_path, 'w', encoding='utf-8') as f_report:
        # 写入表头
        f_report.write(f"{'PatientID':<25} | {'Vertebra':<10} | {'Depth(Z)':<10} | {'Height(Y)':<10} | {'Width(X)':<10} | {'Status'}\n")
        f_report.write("-" * 90 + "\n")

        for filename in tqdm(file_list):
            # --- 解析 Patient ID ---
            if filename.endswith('.nii.gz'):
                subject_name = filename[:-7]
            elif filename.endswith('.nii'):
                subject_name = filename[:-4]
            else:
                subject_name = filename

            # --- 构造路径 ---
            # 假设原始 mask 文件名固定
            mode = prefix_mode
            mask_path = os.path.join(mask_root_dir, subject_name, f"{mode}_processed_mask.nii.gz")
            
            if not os.path.exists(mask_path):
                missing_files.append(subject_name)
                f_report.write(f"{subject_name:<25} | {'ALL':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'MISSING FILE'}\n")
                continue

            try:
                # 读取原始 Mask
                mask_img = sitk.ReadImage(mask_path)
                # 如果需要保存，创建该病人的文件夹
                if save_masks:
                    patient_out_dir = os.path.join(output_root_dir, subject_name)
                    os.makedirs(patient_out_dir, exist_ok=True)

                # --- 遍历每一节椎骨 ---
                for lname, lidx in labels_dict.items():
                    # >>> 核心调用 <<<
                    cleaned_mask, dims_mm = clean_and_measure_vertebra(mask_img, lidx)
                    
                    if cleaned_mask is not None:
                        d, h, w = dims_mm
                        all_dims_mm.append(dims_mm)
                        
                        # 1. 写入报告文件
                        f_report.write(f"{subject_name:<25} | {lname:<10} | {d:<10.2f} | {h:<10.2f} | {w:<10.2f} | {'OK'}\n")
                        
                        # 2. 保存清洗后的 Mask (可选)
                        if save_masks:
                            save_name = f"{lname}_{mode}_mask_cleaned.nii.gz"
                            sitk.WriteImage(cleaned_mask, os.path.join(patient_out_dir, save_name))
                    else:
                        # 记录缺失的椎骨
                        f_report.write(f"{subject_name:<25} | {lname:<10} | {'0.00':<10} | {'0.00':<10} | {'0.00':<10} | {'EMPTY/NO LABEL'}\n")

            except Exception as e:
                print(f"❌ Error: {subject_name} - {e}")
                f_report.write(f"{subject_name:<25} | {'ERROR':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {str(e)}\n")

    # --- 全局统计与建议 ---
    if len(all_dims_mm) > 0:
        all_dims_mm = np.array(all_dims_mm)
        p99_dims = np.percentile(all_dims_mm, 99, axis=0) # [D, H, W]
        
        # 推荐 Crop 计算
        margin_mm = 10.0
        suggested_mm = p99_dims + margin_mm
        
        def round_up_16(x): return int(16 * np.ceil(x / 16))
        # 假设重采样到 1.0mm Spacing
        suggested_px = [round_up_16(x) for x in suggested_mm]

        print("\n" + "="*60)
        print("📊 全局统计报告 (Global Statistics)")
        print("="*60)
        print(f"总计处理椎骨数量: {len(all_dims_mm)}")
        print(f"99% 分位点尺寸 (mm): Depth={p99_dims[0]:.2f}, Height={p99_dims[1]:.2f}, Width={p99_dims[2]:.2f}")
        print("-" * 60)
        print(f"💡 推荐 Crop Box (99% + 10mm Margin, 1.0mm Spacing):")
        print(f"   Shape [Z, Y, X]: {suggested_px}")
        print("="*60)
        print(f"详细数据已保存至: {report_file_path}")
    else:
        print("❌ 未提取到任何有效数据。")

# ================= 🚀 配置区 =================
if __name__ == "__main__":
    # 1. 基础路径配置
    BASE_DIR = "D:/data_space/Zhongrifriendly/paired_data_cropped_176_1/"
    
    # 输入
    IMAGE_ROOT = os.path.join(BASE_DIR, "processed_176_1_volume")
    MASK_ROOT  = os.path.join(BASE_DIR, "processed_176_1_mask")
    
    # 输出
    # 如果开启 SAVE_MASKS，清洗后的 L1-L5 mask 会保存在这里
    OUTPUT_MASK_ROOT = os.path.join(BASE_DIR, "separate_masks_cleaned")
    # 统计报告保存位置
    REPORT_PATH = "./dataset_vertebra_stats.txt"

    # 2. 功能开关
    SAVE_MASKS = True  # <--- 设置为 True 以保存拆分且清洗后的 Mask

    # 3. Label 映射
    LABELS = {
        'L1': 37, 
        'L2': 36,
        'L3': 35,
        'L4': 34,
        'L5': 33
    }
    prefix_mode = 'intra'
    # 4. 执行
    run_dataset_processing(
        IMAGE_ROOT, 
        MASK_ROOT, 
        OUTPUT_MASK_ROOT, 
        REPORT_PATH, 
        LABELS, 
        save_masks=SAVE_MASKS,
        prefix_mode=prefix_mode
    )