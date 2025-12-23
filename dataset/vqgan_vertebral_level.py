
import torch
from torch.utils.data.dataset import Dataset
import os
import random
import glob
import torchio as tio
import json
import random
import pdb
class VQGAN_Vertebral_Dataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        augmentation=False,
        split="train",
        files_names_path=None,
        window_min=-250,
        window_max=2000,
    ):
        randnum = 216
        self.root_dir = root_dir
        print(self.root_dir)

        with open(files_names_path, 'r') as f:
            self.file_names = f.readlines()
        self.file_names = [line.strip() for line in self.file_names]

        self.all_vertebral_level = ["L1", "L2", "L3", "L4", "L5"]
        self.all_vertebral_level_path = []

        for file_name in self.file_names:
            case_path = os.path.join(self.root_dir, file_name)
            for level in self.all_vertebral_level:
                intra_vert_level_path = os.path.join(case_path, level, "intra_aligned.nii.gz")
                pre_vert_level_path = os.path.join(case_path, level, "pre_aligned.nii.gz")
                self.all_vertebral_level_path.append(intra_vert_level_path)
                self.all_vertebral_level_path.append(pre_vert_level_path)
        
        print(f"Total vertebral level paths: {len(self.all_vertebral_level_path)}")
        random.seed(randnum)
        random.shuffle(self.all_vertebral_level_path )

        self.split = split
        self.augmentation = augmentation

        self.randomflip = tio.RandomFlip(axes=(0,1),flip_probability=0.5)

        # 窗宽窗位 / 强度范围配置
        self.window_min = window_min
        self.window_max = window_max

    
    def __len__(self):
        return len(self.all_vertebral_level_path)

    def __getitem__(self, index):
        image_path = self.all_vertebral_level_path[index]
        #pdb.set_trace()
        case_name = image_path.split("\\")[-3].split("/")[-1]
        level = image_path.split("\\")[-2].split("/")[-1]
        type = image_path.split("\\")[-1].split(".")[0]
        whole_img = tio.ScalarImage(image_path)
        img = whole_img  # 保持 ScalarImage 对象以保留 affine 信息
        if self.augmentation:
            img = self.randomflip(img)
        imageout = img.data  # 从 ScalarImage 中提取数据
        if self.augmentation and random.random()>0.5:
            imageout = torch.rot90(imageout,dims=(1,2))

        # 窗宽窗位 / 强度裁剪后归一化到 [-1,1]
        clip_min, clip_max = float(self.window_min), float(self.window_max)

        imageout = torch.clamp(imageout, min=clip_min, max=clip_max)
        print(f"clip_min: {clip_min}, clip_max: {clip_max}")
        print(f"imageout.min(): {imageout.min()}, imageout.max(): {imageout.max()}")
        imageout = (imageout - clip_min) / (clip_max - clip_min + 1e-6)
        imageout = imageout * 2 - 1
        #imageout = imageout.transpose(1,3).transpose(2,3)
        imageout = imageout.type(torch.float32)
         
        if self.split =='val':
            return {'data': imageout, 'affine': img.affine, 'path': image_path , 'level': level , 'names': case_name , 'type': type }
        else:
            return {'data': imageout, 'level': level , 'path': image_path , 'names': case_name , 'type': type}
