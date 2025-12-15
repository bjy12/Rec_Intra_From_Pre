

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import numpy as np
import json 
import torchio as tio


class Res_128_dataset(Dataset):
    def __init__(self, root_dir=None,files_names_path=None, resolution= [32,32,32], generate_latents= False):
        self.root_dir = root_dir 
        print(" dataset_root: " , self.root_dir)
        self.resolution = resolution
        self.generate_latents = generate_latents
        with open(files_names_path, 'r') as f:
            self.file_names = f.readlines()
        self.file_names = [line.strip() for line in self.file_names]
        print(" number of files: " , len(self.file_names))


    def __len__(self):
        return len(self.file_names)


    def __getitem__(self, index):
        file_name = self.file_names[index]
        file_path = os.path.join(self.root_dir, f"{file_name}.nii.gz")
        if self.generate_latents:
            img = tio.ScalarImage(file_path)
            img_data = img.data.to(torch.float32)
            imageout = img_data * 2 - 1
            imageout = imageout.transpose(1,3).transpose(2,3)
            return imageout, file_name
        else:
            latent_path = os.path.join(self.root_dir, f"latent_{file_name}.nii.gz")
            latent = tio.ScalarImage(latent_path)
            latent = latent.data.to(torch.float32)
            # 单类别数据集：所有样本的类别标签都设为 0
            cls_idx = 0
            return latent, torch.tensor(int(cls_idx)), torch.tensor(self.resolution)/64.0