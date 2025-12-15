

import numpy as np
from sympy import Q
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import numpy as np
import json 
import torchio as tio
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pickle
import yaml
from dataset.geometry import Geometry
from copy import deepcopy

import pdb

class Pre_Intra_Dataset_Ver_128(Dataset):
    def __init__(self, root_dir=None,files_names_path=None , geo_config_path=None ):
        self.root_dir = root_dir 

        self.drr_dir = os.path.join(self.root_dir,'projections')
        print(" ct_root: " , self.root_dir)
        print(" drr_root: " , self.drr_dir)
        with open(files_names_path, 'r') as f:
            self.file_names = f.readlines()
        self.file_names = [line.strip() for line in self.file_names]
        print(" number of files: " , len(self.file_names))
        with open(geo_config_path, 'r') as f:
            geo_config =  yaml.safe_load(f)
        self.geometry = Geometry(geo_config['projector'])
        self.low_res_points = self.create_low_res_space(geo_config['projector']['nVoxel'])

    def __len__(self):
        return len(self.file_names)

    def project_points(self, points , angles):
        points_proj = []
        for a in angles:
            p = self.geometry.project(points , a )
            points_proj.append(p)
        points_proj = np.stack(points_proj , axis=0)
        return points_proj

    def create_low_res_space(self , low_res):
        x, y, z = np.meshgrid(
            np.arange(low_res[0]),
            np.arange(low_res[1]),
            np.arange(low_res[2]),
            indexing='ij'
        )
        #pdb.set_trace()
        coords = np.stack([
            x.flatten(),
            y.flatten(),
            z.flatten()
        ], axis=1)

        # 归一化到 [0,1] 范围
        coords = coords / (np.array(low_res) - 1)
        #pdb.set_trace()
        # 添加batch维度，转换为float32类型
        coords = coords.reshape(-1, 3).astype(np.float32)

        return coords

    def sample_projections(self, name, n_view=2):
        # -- load projections
        with open(os.path.join(self.root_dir, 'projections', f"{name}.pickle"), 'rb') as f:
            data = pickle.load(f)
            projs = data['projs']         # uint8: [K, W, H]
            projs_max = data['projs_max'] # float
            angles = data['angles']       # float: [K,]

        if n_view is None:
            n_view = self.num_views
        #pdb.set_trace()
        # -- sample projections
        views = np.linspace(0, len(projs), n_view, endpoint=False).astype(int) # endpoint=False as the random_views is True during training, i.e., enabling view offsets.
        projs = projs[views].astype(np.float32) / 255.
        projs = projs[:, None, ...]
        angles = angles[views]
        # normalization to [-1 , 1]
        projs = (projs * 2) - 1
        # -- de-normalization
        #projs = projs * projs_max / 0.2

        return projs, angles     
        

    def __getitem__(self, index):
        file_name = self.file_names[index]
        name = file_name.split('_')[0]
        # pre or intra
        pre_file_name = f"{name}_pre_processed_volume.nii.gz"
        intra_file_name = f"{name}_intra_processed_volume.nii.gz"
        pre_ct_path = os.path.join(self.root_dir,'images', f"{file_name}.nii.gz")
        intra_ct_path = os.path.join(self.root_dir,'images', f"{file_name}.nii.gz")

        pre_ct = tio.ScalarImage(pre_ct_path)
        intra_ct = tio.ScalarImage(intra_ct_path)
        pre_ct = pre_ct.data.to(torch.float32)
        intra_ct = intra_ct.data.to(torch.float32)
        # normalize 
        pre_ct = pre_ct * 2 - 1
        intra_ct = intra_ct * 2 - 1 
        # 
        pre_latent_path = os.path.join(self.root_dir, 'latent_ds', f"latent_{name}_pre_processed_volume.nii.gz")
        intra_latent_path = os.path.join(self.root_dir, 'latent_ds', f"latent_{name}_intra_processed_volume.nii.gz")
        pre_latent = tio.ScalarImage(pre_latent_path)
        intra_latent = tio.ScalarImage(intra_latent_path)
        pre_latent = pre_latent.data.to(torch.float32)
        intra_latent = intra_latent.data.to(torch.float32)

        #pdb.set_trace()
        projs, angles = self.sample_projections(file_name, n_view=2)
        #pdb.set_trace()
        proj_points = self.project_points(self.low_res_points, angles)
        
        projs = torch.from_numpy(projs).to(torch.float32)
        proj_points = torch.from_numpy(proj_points).to(torch.float32)

        points = deepcopy(self.low_res_points)
        points[:, :2] -= 0.5  
        points[:, 2]  = 0.5 - points[:,2]
        points *= 2 



        output = {
            'pre_ct': pre_ct,
            'intra_ct': intra_ct,
            'name': name,
            'projs': projs,
            'angles': angles,
            'pre_latent': pre_latent,
            'intra_latent': intra_latent,
            'coords': points,
            'proj_points': proj_points,
        }

        return output







        latent_path = os.path.join(self.root_dir, f"latent_{file_name}.nii.gz")
        latent = tio.ScalarImage(latent_path)
        latent = latent.data.to(torch.float32)
        # 单类别数据集：所有样本的类别标签都设为 0
        cls_idx = 0
        return latent, torch.tensor(int(cls_idx)), torch.tensor(self.resolution)/64.0

def visulize_load_ct_from_sitk(ct):
    h , w , d = ct.shape
    slice_0 = ct[h//2 , : , :]
    slice_1 = ct[: , w//2 , :]
    slice_2 = ct[: , : , d//2]
    # use matplotlib to visualize the slice_0, slice_1, slice_2
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(slice_0, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(slice_1, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(slice_2, cmap='gray')
    plt.show()


def visulize_load_ct_from_tio(ct):
    image = ct.data.numpy()
    image = image[0]
    h , w , d = image.shape
    slice_0 = image[h//2 , : , :]
    slice_1 = image[: , w//2 , :]
    slice_2 = image[: , : , d//2]
    # use matplotlib to visualize the slice_0, slice_1, slice_2
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(slice_0, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(slice_1, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(slice_2, cmap='gray')
    plt.show()      


#main 
if __name__ == "__main__":
    dataset = Pre_Intra_Dataset_Ver_128(root_dir='D:/data_space/Zhongrifriendly/paired_process_128_tigre', 
                                        files_names_path='./files_names/train_files.txt',
                                        geo_config_path='D:/code_space_bone/3D-MedDiffusion-main/3D-MedDiffusion-main/geo_config/config_128_1_25_ls.yaml')
    d_0 = dataset[0]
    pdb.set_trace()
    print(" d_0 keys : " , d_0.keys())
    print(" d_0 pre_ct shape : " , d_0['pre_ct'].shape)
    print(" d_0 intra_ct shape : " , d_0['intra_ct'].shape)
    print(" d_0 name : " , d_0['name'])
    print(" d_0 projs shape : " , d_0['projs'].shape)
    print(" d_0 angles shape : " , d_0['angles'].shape)
    print(" d_0 pre_latent shape : " , d_0['pre_latent'].shape)
    print(" d_0 intra_latent shape : " , d_0['intra_latent'].shape)
