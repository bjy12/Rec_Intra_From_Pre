

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from torch.utils.data import DataLoader
from AutoEncoder.model.PatchVolume import patchvolumeAE 
from dataset.Singleres_dataset import Singleres_dataset
from dataset.Singleres_dataset_ver_128 import Res_128_dataset
from dataset.vqgan_vertebral_level import VQGAN_Vertebral_Dataset
import torch
from os.path import join
import argparse
import torchio as tio
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

import pdb

def generate():
    root_dir = 'D:/data_space/Zhongrifriendly/paired_data_cropped_176_1/final_dataset/'
    files_names_path = './files_names/test_cases_vertebral_ds.txt'
    latent_ds_save_root = 'D:/data_space/Zhongrifriendly/paired_data_cropped_176_1/latent_ds/'
    batch_size = 1
    num_workers = 1
    AE_ckpt = 'D:/code_space_bone/3D-MedDiffusion-main/my_model/latest_checkpoint-v2.ckpt'
    #tr_dataset = Singleres_dataset(root_dir=args.data_path,generate_latents = True)
    #tr_dataset = Res_128_dataset(root_dir=root_dir,files_names_path=files_names_path,generate_latents = True)
    tr_dataset = VQGAN_Vertebral_Dataset(root_dir = root_dir , 
                                         augmentation = False , split = 'val' , files_names_path = files_names_path,
                                         window_min = -250 , window_max = 2000)
    #td_0 = tr_dataset[0]
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
    #pdb.set_trace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AE = patchvolumeAE.load_from_checkpoint(AE_ckpt)
    AE = AE.to(device)
    AE.eval()
    #pdb.set_trace()
    if not os.path.exists(latent_ds_save_root):
        os.makedirs(latent_ds_save_root)
    
    for batch in tr_dataloader:

        #pdb.set_trace()
        sample = batch['data']
        sample = sample.cuda()
        with torch.no_grad():
            #z =  AE.patch_encode(sample,patch_size = 64)
            z = AE.encode(sample,quantize=False)
            #pdb.set_trace()
            output = ((z - AE.codebook.embeddings.min()) /
            (AE.codebook.embeddings.max() -
            AE.codebook.embeddings.min())) * 2.0 - 1.0
        output = output.cpu()
        #pdb.set_trace()
        output_ = output[0]
        case_name = batch['names'][0]
        level = batch['level'][0]
        type = batch['type'][0]
        affine = batch['affine'][0]
        #pdb.set_trace()
        out_put_path = os.path.join(latent_ds_save_root, f"lt_{case_name}_{level}_{type}.nii.gz")
        img = tio.ScalarImage(tensor=output_,affine=affine)
        img.save(out_put_path)   

if __name__ == "__main__":
    generate()



