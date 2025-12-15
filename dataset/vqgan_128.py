
import torch
from torch.utils.data.dataset import Dataset
import os
import random
import glob
import torchio as tio
import json
import random

class VQGANDataset_128_full_CT(Dataset):
    def __init__(self, root_dir=None, augmentation=False,split='train', files_names_path=None):
        randnum = 216
        self.root_dir = root_dir
        print(self.root_dir)

        with open(files_names_path, 'r') as f:
            self.file_names = f.readlines()
        self.file_names = [line.strip() for line in self.file_names]

        random.seed(randnum)
        random.shuffle(self.file_names )

        self.split = split
        self.augmentation = augmentation

        self.randomflip = tio.RandomFlip( axes=(0,1),flip_probability=0.5)

    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_path = os.path.join(self.root_dir, f'{file_name}.nii.gz')
        whole_img = tio.ScalarImage(image_path)
        img = whole_img  # 保持 ScalarImage 对象以保留 affine 信息
        if self.augmentation:
            img = self.randomflip(img)
        imageout = img.data  # 从 ScalarImage 中提取数据
        if self.augmentation and random.random()>0.5:
            imageout = torch.rot90(imageout,dims=(1,2))
            
        imageout = imageout * 2 - 1
        imageout = imageout.transpose(1,3).transpose(2,3)
        imageout = imageout.type(torch.float32)

        if self.split =='val':
            return {'data': imageout, 'affine': img.affine, 'path': file_name}
        else:
            return {'data': imageout}
