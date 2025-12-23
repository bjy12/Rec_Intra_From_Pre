"""
Pre_Intra_Final_Dataset: Dataset loader for multi-task condition branch training.

加载 final_dataset 结构的数据：
- pre_ct_masked.nii.gz: 术前 CT
- pre_aligned.nii.gz: 对齐到术中空间的 CT (配准 GT)
- intra_aligned.nii.gz: 术中 CT (生成 GT) 
- Level_x_roi.pkl: 包含两张 DRR 图像
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import yaml
from dataset.geometry import Geometry
from monai.transforms import Resize
import pdb

class RandomRigidTransform3D:
    """
    对 3D 体积应用随机刚体变换 (旋转 + 平移)，并返回变换参数。
    变换参数归一化到 [-1, 1] 范围，便于网络预测。
    """
    def __init__(
        self,
        max_rotation_deg: float = 10.0,
        max_translation_voxels: float = 5.0,
        volume_size: Tuple[int, int, int] = (128, 128, 128),
        fill_value: float = 0.0,
    ):
        self.max_rotation_rad = np.deg2rad(max_rotation_deg)
        self.max_translation = max_translation_voxels
        self.volume_size = volume_size
        self.fill_value = float(fill_value)
    
    def __call__(self, volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            volume: [1, D, H, W] tensor
        Returns:
            transformed_volume: [1, D, H, W]
            transform_params: [6] normalized parameters (rx, ry, rz, tx, ty, tz)
        """
        # Sample random rotation angles (in radians)
        rx = np.random.uniform(-self.max_rotation_rad, self.max_rotation_rad)
        ry = np.random.uniform(-self.max_rotation_rad, self.max_rotation_rad)
        rz = np.random.uniform(-self.max_rotation_rad, self.max_rotation_rad)
        
        # Sample random translations (in voxels, then normalize to [-1, 1])
        tx = np.random.uniform(-self.max_translation, self.max_translation)
        ty = np.random.uniform(-self.max_translation, self.max_translation)
        tz = np.random.uniform(-self.max_translation, self.max_translation)
        
        # Build rotation matrices
        Rx = torch.tensor([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ], dtype=torch.float32)
        
        Ry = torch.tensor([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ], dtype=torch.float32)
        
        Rz = torch.tensor([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        R = Rz @ Ry @ Rx  # Combined rotation matrix
        
        # Normalize translation to [-1, 1] for grid_sample
        # grid_sample expects coordinates in [-1, 1]
        D, H, W = self.volume_size
        tx_norm = tx / (W / 2)
        ty_norm = ty / (H / 2)
        tz_norm = tz / (D / 2)
        
        # Build affine matrix [3, 4]
        affine = torch.zeros(3, 4, dtype=torch.float32)
        affine[:3, :3] = R
        affine[0, 3] = tx_norm
        affine[1, 3] = ty_norm
        affine[2, 3] = tz_norm
        
        # Apply transformation using grid_sample
        # volume shape: [1, D, H, W] -> [1, 1, D, H, W] for 3D grid_sample
        vol = volume.unsqueeze(0)  # [1, 1, D, H, W]
        # shift by fill_value so that out-of-bounds zeros map back to desired constant
        vol = vol - self.fill_value
        
        grid = F.affine_grid(affine.unsqueeze(0), vol.shape, align_corners=True)
        transformed = F.grid_sample(
            vol, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )
        transformed = transformed + self.fill_value
        transformed = transformed.squeeze(0)  # [1, D, H, W]
        
        # Normalize parameters to [-1, 1] for network prediction
        params = torch.tensor([
            rx / self.max_rotation_rad,
            ry / self.max_rotation_rad,
            rz / self.max_rotation_rad,
            tx / self.max_translation,
            ty / self.max_translation,
            tz / self.max_translation
        ], dtype=torch.float32)
        
        return transformed, params


class Pre_Intra_Final_Dataset(Dataset):
    """
    Dataset for multi-task condition branch training.
    
    Args:
        root_dir: Path to final_dataset directory
        files_names_path: Path to txt file with case_name/level entries
        geo_config_path: Path to geometry config yaml
        max_rotation_deg: Maximum rotation for random perturbation
        max_translation_voxels: Maximum translation for random perturbation
        apply_perturbation: Whether to apply random perturbation (disable for validation)
    """
    def __init__(
        self,
        root_dir: str,
        files_names_path: str,
        geo_config_path: str,
        max_rotation_deg: float = 20.0,
        max_translation_voxels: float = 10.0,
        apply_perturbation: bool = True,
        window_min: float = -250,
        window_max: float = 2000,
    ):
        self.root_dir = root_dir
        self.apply_perturbation = apply_perturbation
        
        # Load file names
        with open(files_names_path, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
        print(f"[Pre_Intra_Final_Dataset] Loaded {len(self.file_names)} samples from {files_names_path}")

        self.all_vertebral_level = ["L1", "L2", "L3", "L4", "L5"]
        self.all_vertebral_level_path = []
        for file_name in self.file_names:
            case_path = os.path.join(self.root_dir, file_name)
            for level in self.all_vertebral_level:
                level_path = os.path.join(case_path, level)
                self.all_vertebral_level_path.append(level_path)
        print(f"Total vertebral level paths: {len(self.all_vertebral_level_path)}")
        #print(f"Vertebral level paths: {self.all_vertebral_level_path}")

        self.window_min = window_min
        self.window_max = window_max


        self.resize_transform = Resize(spatial_size=(128, 128), mode='bilinear')  # Resize DRR to 128x128
        self.ct_resize_transform = Resize(spatial_size=(128, 128, 128), mode='trilinear')  # Resize CT to 128³
        self.target_volume_size = (128, 128, 128)  # Model expects 128³

        
        # Load geometry config
        with open(geo_config_path, 'r') as f:
            geo_config = yaml.safe_load(f)
        self.geometry = Geometry(geo_config['projector'])
        self.volume_size = tuple(geo_config['projector']['nVoxel'])
        
        # Create low-res space for projection points
        self.low_res_points = self._create_low_res_space(self.volume_size)
        
        # Random transform augmentation - use target volume size (128³)
        self.random_transform = RandomRigidTransform3D(
            max_rotation_deg=max_rotation_deg,
            max_translation_voxels=max_translation_voxels,
            volume_size=self.target_volume_size,
            fill_value = -1.0
        )
    
    def _create_low_res_space(self, res: Tuple[int, int, int]) -> np.ndarray:
        """Create normalized [0,1] coordinate grid for projection."""
        x, y, z = np.meshgrid(
            np.arange(res[0]),
            np.arange(res[1]),
            np.arange(res[2]),
            indexing='ij'
        )
        coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        coords = coords / (np.array(res) - 1)
        return coords.astype(np.float32)
    
    def _project_points(self, points: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D for each view angle."""
        points_proj = []
        for a in angles:
            p = self.geometry.project(points, a)
            points_proj.append(p)
        return np.stack(points_proj, axis=0)
    
    def _load_drr_from_pkl(self, pkl_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load DRR images from pkl file.
        Expected format: dict with 'drr_ap', 'drr_lat' or similar keys.
        Returns: (drr_images [2, H, W], angles [2])
        """
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        #pdb.set_trace()
        roi_list =  data['roi'] #  each item is a numpy array
        resize_roi_list = []
        for roi in roi_list:
            # roi is np array
            roi = roi.astype(np.float32)
            roi = roi / 255. 
            roi = roi * 2 - 1  
            #pdb.set_trace()
            roi = torch.from_numpy(roi).unsqueeze(0)
            #pdb.set_trace()
            roi = self.resize_transform(roi)
            resize_roi_list.append(roi)
        resized_rois = torch.stack(resize_roi_list, dim=0)
        #pdb.set_trace()
        angles = data['angles']
        return resized_rois, angles
    
    def __len__(self) -> int:
        return len(self.all_vertebral_level_path)
    def normalize_ct(self, ct: torch.Tensor) -> torch.Tensor:
        clip_min, clip_max = float(self.window_min), float(self.window_max)
        ct = torch.clamp(ct, min=clip_min, max=clip_max)
        ct = (ct - clip_min) / (clip_max - clip_min + 1e-6)
        ct = ct * 2 - 1
        return ct

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        level_case_path = self.all_vertebral_level_path[index]
        #pdb.set_trace()
        level = level_case_path.split("\\")[-1]
        case_name = level_case_path.split("\\")[-2]
        pre_aligned_path = os.path.join(level_case_path, "pre_aligned.nii.gz")
        intra_aligned_path = os.path.join(level_case_path, "intra_aligned.nii.gz")
        drr_path = os.path.join(level_case_path, f"Level_{level}_roi.pkl")
        pre_aligned = tio.ScalarImage(pre_aligned_path).data.to(torch.float32)  # [1, D, H, W]
        intra_aligned = tio.ScalarImage(intra_aligned_path).data.to(torch.float32)
        
        # Resize CT to 128³ if needed
        if pre_aligned.shape[1:] != self.target_volume_size:
            pre_aligned = self.ct_resize_transform(pre_aligned)
            intra_aligned = self.ct_resize_transform(intra_aligned)
        
        # Normalize CT to [-1, 1]
        pre_aligned = self.normalize_ct(pre_aligned)
        intra_aligned = self.normalize_ct(intra_aligned)
        #pdb.set_trace()
        # Apply random perturbation to pre_aligned
        if self.apply_perturbation:
            perturbed_ct, transform_params = self.random_transform(pre_aligned)
        else:
            perturbed_ct = pre_aligned.clone()
            transform_params = torch.zeros(6, dtype=torch.float32)
        
        # Load DRR images
        drr_images, angles = self._load_drr_from_pkl(drr_path)
        
        return {
            'perturbed_ct': perturbed_ct,           # [1, D, H, W] - Network input
            'pre_aligned': pre_aligned,              # [1, D, H, W] - Registration GT
            'intra_ct': intra_aligned,                    # [1, D, H, W] - Generation GT
            'drr_images': drr_images,                # [2, 1, H, W] - DRR input
            'transform_params': transform_params,    # [6] - Registration GT params
            'name': case_name,
            'level': level,
            'angles': angles,
        }


# Test code
if __name__ == "__main__":
    
    # Example usage
    dataset = Pre_Intra_Final_Dataset(
        root_dir=r"D:\data_space\Zhongrifriendly\paired_data_cropped_176_1\final_dataset",
        files_names_path=r"./files_names/train_cases_vertebral_ds.txt",
        geo_config_path=r"./Dataset_Preprocess/config_pre/config_172_1_600.yaml"
    )
    
    sample = dataset[0]
    # 保存扰动前后 CT 以查看效果
    debug_dir = "./debug_perturb_view"
    os.makedirs(debug_dir, exist_ok=True)
    tio.ScalarImage(tensor=sample['perturbed_ct'], affine=np.eye(4)).save(
        os.path.join(debug_dir, "perturbed_ct.nii.gz")
    )
    tio.ScalarImage(tensor=sample['pre_aligned'], affine=np.eye(4)).save(
        os.path.join(debug_dir, "pre_aligned_ct.nii.gz")
    )
    tio.ScalarImage(tensor=sample['intra_ct'], affine=np.eye(4)).save(
        os.path.join(debug_dir, "intra_ct.nii.gz")
    )
    pdb.set_trace()
    print("Sample keys:", sample.keys())
    print("perturbed_ct shape:", sample['perturbed_ct'].shape)
    print("pre_aligned shape:", sample['pre_aligned'].shape)
    print("intra_ct shape:", sample['intra_ct'].shape)
    print("drr_images shape:", sample['drr_images'].shape)
    print("transform_params:", sample['transform_params'])
    print("name:", sample['name'])
    print("level:", sample['level'])
    print("angles:", sample['angles'])

    #* save per
