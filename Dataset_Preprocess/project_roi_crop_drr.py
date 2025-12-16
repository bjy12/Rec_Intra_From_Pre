import json
import os
import pickle

import matplotlib.colors as mcolors  # 保留以便可视化叠加
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tigre
import yaml
from tigre.utilities.Ax import Ax

# ==========================================
# 1. 基础配置与函数
# ==========================================


def sitk_load_luna16(path):
    itk_img = sitk.ReadImage(path)
    origin = np.array(itk_img.GetOrigin(), dtype=np.float32)
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)
    image = sitk.GetArrayFromImage(itk_img)
    # SimpleITK 读取顺序为 (z, y, x)，转换为 (x, y, z) 方便后续处理
    image = image.transpose(2, 1, 0)
    image = image.astype(np.float32)
    return image, spacing, origin


class ConeGeometry_special(tigre.utilities.geometry.Geometry):
    def __init__(self, config):
        super().__init__()
        self.DSD = config["DSD"] / 1000
        self.DSO = config["DSO"] / 1000
        self.nDetector = np.array(config["nDetector"])
        self.dDetector = np.array(config["dDetector"]) / 1000
        self.sDetector = self.nDetector * self.dDetector
        # 初始占位，后续会被真实数据覆盖
        self.nVoxel = np.array(config["nVoxel"][::-1])
        self.dVoxel = np.array(config["dVoxel"][::-1]) / 1000
        self.sVoxel = self.nVoxel * self.dVoxel
        self.offOrigin = np.array(config["offOrigin"][::-1]) / 1000
        self.offDetector = np.array(
            [config["offDetector"][1], config["offDetector"][0], 0]
        ) / 1000
        self.accuracy = config["accuracy"]
        self.mode = config["mode"]


def percentile_normalize(img, low=1, high=99):
    """将影像归一化到 0-1（鲁棒分位数），避免极端值影响显示。"""
    vmin, vmax = np.percentile(img, [low, high])
    return np.clip((img - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)


def load_drr_from_pickle(path):
    """读取 pickle 中的 DRR 投影，兼容常见结构。"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    projections = None
    angles_in_file = None
    #import pdb
    #pdb.set_trace()
    if isinstance(data, dict):
        for key in ["projections", "projs", "drr", "data"]:
            if key in data:
                projections = data[key]
                break
        for key in ["angles", "thetas", "theta", "angle_rad", "angle"]:
            if key in data:
                angles_in_file = np.asarray(data[key], dtype=np.float32)
                # 如果是度数则转为弧度
                if angles_in_file.max() > 2 * np.pi:
                    angles_in_file = np.deg2rad(angles_in_file)
                break
    else:
        projections = data

    if projections is None:
        raise ValueError(f"无法从 {path} 解析出投影数组")

    projections = np.asarray(projections)
    if projections.ndim == 2:
        projections = projections[None, ...]

    return projections, angles_in_file


def project_label(mask_vol, geo, angles_rad, flip_detector=True):
    """前向投影单标签体，返回 [n_angles, H, W]。"""
    proj = Ax(mask_vol, geo, angles_rad, "interpolated")
    if flip_detector:
        proj = proj[:, ::-1, :]
    return proj


def compute_centroid_and_bbox(proj, threshold=1e-6, margin=10, clamp_shape=None):
    """根据投影 mask 计算质心与 bbox（含边界检查）。"""
    mask = proj > threshold
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None

    weights = proj[ys, xs]
    cy = float(np.sum(ys * weights) / (weights.sum() + 1e-8))
    cx = float(np.sum(xs * weights) / (weights.sum() + 1e-8))

    y0 = int(np.floor(ys.min()) - margin)
    y1 = int(np.ceil(ys.max()) + margin + 1)
    x0 = int(np.floor(xs.min()) - margin)
    x1 = int(np.ceil(xs.max()) + margin + 1)

    if clamp_shape is not None:
        h, w = clamp_shape
        y0 = max(0, y0)
        x0 = max(0, x0)
        y1 = min(h, y1)
        x1 = min(w, x1)

    return (cy, cx), (y0, y1, x0, x1)


def save_overlay(drr_norm, proj_norm, bbox, centroid, save_path, title=""):
    """在 DRR 上叠加投影与 bbox，便于核查。"""
    plt.figure(figsize=(6, 6))
    plt.imshow(drr_norm, cmap="gray")
    # 叠加标签投影等高线
    plt.contour(proj_norm, levels=[0.3, 0.6, 0.9], colors="lime", linewidths=1)
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        plt.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            "r-",
            linewidth=2,
        )
    if centroid is not None:
        plt.plot(centroid[1], centroid[0], "yo", markersize=6)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==========================================
# 2. 路径与超参（批量模式）
# ==========================================
VOLUME_ROOT = (
    r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\processed_176_1_volume"
)
LABEL_ROOT = (
    r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\cropped_by_mask"
)
DRR_ROOT = (
    r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\tigre_process_1\projections"
)
CONFIG_PATH = "./Dataset_Preprocess/config_pre/config_172_1_600.yaml"
OUT_ROOT = "D:/data_space/Zhongrifriendly/paired_data_cropped_176_1/label_drr_roi_600"
os.makedirs(OUT_ROOT, exist_ok=True)

# 默认角度（若 pickle 中未携带角度信息则回退到这里）
ANGLES_DEG_DEFAULT = [0, 90]
# bbox 额外边距（像素）
MARGIN_PX = 1
# 是否翻转探测器轴以匹配存储的 DRR 方向
FLIP_DETECTOR = True
# label 子文件夹名单（可按需增减；为空则自动使用发现的全部）
LABEL_FOLDERS = ["L1", "L2", "L3", "L4", "L5"]

# ==========================================
# 3. 主流程：批量投影 label → bbox → 裁剪 DRR
# ==========================================
if __name__ == "__main__":
    # 读取配置模板
    with open(CONFIG_PATH, "r") as f:
        config_template = yaml.safe_load(f)

    # 发现 case 列表（以 LABEL_ROOT 下的子目录为准）
    case_names = sorted(
        [
            d
            for d in os.listdir(LABEL_ROOT)
            if os.path.isdir(os.path.join(LABEL_ROOT, d))
        ]
    )
    print(f"Found {len(case_names)} cases under label root.")

    for case in case_names:
        print(f"\n=== Case: {case} ===")
        ct_path = os.path.join(VOLUME_ROOT, case, "intra_processed_volume.nii.gz")
        drr_path = os.path.join(DRR_ROOT, f"{case}_intra_processed_volume.pickle")
        case_label_root = os.path.join(LABEL_ROOT, case)
        case_out_dir = os.path.join(OUT_ROOT, case)
        os.makedirs(case_out_dir, exist_ok=True)

        if not os.path.exists(ct_path):
            print(f"[Warn] CT not found: {ct_path}, skip.")
            continue
        if not os.path.exists(drr_path):
            print(f"[Warn] DRR pickle not found: {drr_path}, skip.")
            continue
        if not os.path.isdir(case_label_root):
            print(f"[Warn] Label folder missing: {case_label_root}, skip.")
            continue

        # 加载 CT
        ct_vol, ct_spacing, _ = sitk_load_luna16(ct_path)
        ct_input = ct_vol.transpose(2, 1, 0).copy()  # (Z, Y, X)

        # 几何配置（覆盖真实体素信息）
        geo = ConeGeometry_special(config_template["projector"])
        geo.nVoxel = np.array(ct_input.shape)
        geo.dVoxel = np.array([ct_spacing[2], ct_spacing[1], ct_spacing[0]]) / 1000
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        print(f"Geometry updated: nVoxel={geo.nVoxel}, sVoxel={geo.sVoxel}")

        # 读取 DRR
        drr_stack, angles_from_file = load_drr_from_pickle(drr_path)
        angles_rad = (
            angles_from_file
            if angles_from_file is not None
            else np.deg2rad(ANGLES_DEG_DEFAULT)
        )
        angles_deg_used = (
            np.rad2deg(angles_rad).tolist()
            if angles_from_file is not None
            else ANGLES_DEG_DEFAULT
        )
        if drr_stack.shape[0] != len(angles_rad):
            print(
                f"[Warn] DRR 数量 {drr_stack.shape[0]} 与角度数 {len(angles_rad)} 不一致，将按最小数量对齐。"
            )

        # 遍历 label 子文件夹
        available_labels = sorted(
            [
                d
                for d in os.listdir(case_label_root)
                if os.path.isdir(os.path.join(case_label_root, d))
            ]
        )
        if LABEL_FOLDERS:
            label_folders = [l for l in LABEL_FOLDERS if l in available_labels]
        else:
            label_folders = available_labels

        case_results = []
        for label_folder in label_folders:
            mask_path = os.path.join(
                case_label_root, label_folder, "mask_crop_fullsize.nii.gz"
            )
            if not os.path.exists(mask_path):
                print(f"[Warn] Mask missing: {mask_path}, skip label.")
                continue

            print(f"Processing label {label_folder} ...")
            label_vol, _, _ = sitk_load_luna16(mask_path)
            label_input = label_vol.transpose(2, 1, 0).copy()  # (Z, Y, X)
            mask_vol = (label_input > 0).astype(np.float32)

            label_proj_all = project_label(
                mask_vol, geo, angles_rad, flip_detector=FLIP_DETECTOR
            )

            num_angles = min(drr_stack.shape[0], label_proj_all.shape[0])
            label_results = []
            roi_list = []
            # 直接放到 case 目录下，文件名前缀标记 Level_x，便于直接查看
            label_out_dir = case_out_dir

            for idx in range(num_angles):
                angle_deg = (
                    angles_deg_used[idx]
                    if idx < len(angles_deg_used)
                    else f"idx_{idx}"
                )
                drr = drr_stack[idx]
                label_proj = label_proj_all[idx]

                centroid, bbox = compute_centroid_and_bbox(
                    label_proj,
                    threshold=1e-6,
                    margin=MARGIN_PX,
                    clamp_shape=drr.shape,
                )

                if bbox is None:
                    print(f"[Warn] 角度 {angle_deg} 未找到标签投影，跳过。")
                    continue

                y0, y1, x0, x1 = bbox
                cropped = drr[y0:y1, x0:x1]
                cropped_norm = percentile_normalize(cropped)
                drr_norm = percentile_normalize(drr)
                label_norm = percentile_normalize(label_proj)
                roi_list.append({"angle": float(angle_deg), "roi": cropped})

                prefix = f"Level_{label_folder}"
                crop_path = os.path.join(
                    label_out_dir, f"{prefix}_crop_angle_{angle_deg}.png"
                )
                overlay_path = os.path.join(
                    label_out_dir, f"{prefix}_overlay_angle_{angle_deg}.png"
                )

                plt.imsave(crop_path, cropped_norm, cmap="gray")
                save_overlay(
                    drr_norm,
                    label_norm,
                    bbox,
                    centroid,
                    overlay_path,
                    title=f"Angle {angle_deg}°",
                )

                label_results.append(
                    {
                        "angle_deg": float(angle_deg)
                        if isinstance(angle_deg, (int, float, np.floating))
                        else angle_deg,
                        "centroid_xy": [float(centroid[1]), float(centroid[0])],
                        "bbox_yx": [int(y0), int(y1), int(x0), int(x1)],
                        "crop_path": crop_path,
                        "overlay_path": overlay_path,
                    }
                )
                print(
                    f"Angle {angle_deg}: bbox(y0,y1,x0,x1)={bbox}, "
                    f"centroid(y,x)=({centroid[0]:.2f}, {centroid[1]:.2f})"
                )

            # 保存 label 元信息
            label_meta_path = os.path.join(
                label_out_dir, f"Level_{label_folder}_bbox_results.json"
            )
            with open(label_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "case": case,
                        "label": label_folder,
                        "angles_deg": angles_deg_used,
                        "results": label_results,
                        "angles_in_file": None
                        if angles_from_file is None
                        else np.rad2deg(angles_from_file).tolist(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            # 保存该 level 的 ROI pickle
            roi_pickle_path = os.path.join(
                label_out_dir, f"Level_{label_folder}_roi.pkl"
            )
            with open(roi_pickle_path, "wb") as f:
                pickle.dump(
                    {
                        "level": label_folder,
                        "angles": [r["angle"] for r in roi_list],
                        "roi": [r["roi"] for r in roi_list],
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            case_results.append(
                {
                    "label": label_folder,
                    "results": label_results,
                    "meta": label_meta_path,
                    "roi_pickle": roi_pickle_path,
                }
            )

        # 保存 case 汇总
        case_meta_path = os.path.join(case_out_dir, "case_bbox_results.json")
        with open(case_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "case": case,
                    "angles_deg": angles_deg_used,
                    "angles_in_file": None
                    if angles_from_file is None
                    else np.rad2deg(angles_from_file).tolist(),
                    "labels": case_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Case {case} done. Results saved to {case_out_dir}")