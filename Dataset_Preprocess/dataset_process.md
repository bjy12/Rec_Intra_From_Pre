# Vertebra-level 数据预处理流水线

1) `resample_image_mask.py`  
   将原始 image 与对应 mask 重采样到目标 spacing / 分辨率。

2) `count_verebra_bbox.py`  
   清洗并统计各 level 脊椎分割，提取 bbox 信息（辅助筛查）。

3) `check_segmask.py`  
   可视化每个病例、每个椎体的中间切片（全标签、逐标签、按 level 叠加），快速检查掩码质量；将不可用的 case 记录到 `black_case.txt`。

4) `crop_roi_with_margin.py`  
   对（未配准的）level 掩码做膨胀并对 CT 进行全尺寸 masking，统计有效 ROI 尺寸；遇到 `black_case.txt` 中的病例会跳过。

5) `run_elastix_batch.py`  
   对每个椎体执行刚性配准（pre→intra），生成 `TransformParameters.0.txt` 和配准产物。

6) `register_premask_2_intract.py`  
   读取配准矩阵，把术前高质量 mask 重采样到术中 CT 空间，得到对齐的术中感兴趣区域（`moving_in_fixed.nii.gz` 等）。

7) `crop_intra_ct_and_registred_pre_ct.py`  
   使用已对齐的 mask 在术中 CT 上批量裁剪各椎体 ROI，输出裁剪体、全尺寸对齐版本和 bbox 统计；同样跳过 `black_case.txt` 中的病例。
