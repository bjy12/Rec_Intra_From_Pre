

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from torch.utils.data import DataLoader
from AutoEncoder.model.PatchVolume import patchvolumeAE 
from dataset.Singleres_dataset import Singleres_dataset
from dataset.Singleres_dataset_ver_128 import Res_128_dataset
import torch
from os.path import join
import argparse
import torchio as tio
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

import pdb

def generate():
    root_dir = 'D:/data_space/Zhongrifriendly/paired_process_128_tigre/images/'
    files_names_path = './files_names/val_files.txt'
    latent_ds_save_root = 'D:/data_space/Zhongrifriendly/paired_process_128_tigre/latent_ds/'
    batch_size = 1
    num_workers = 1
    AE_ckpt = 'D:/code_space_bone/3D-MedDiffusion-main/ver_128_full_VQAE/results/my_model/version_0/checkpoints/latest_checkpoint-v2.ckpt'
    #tr_dataset = Singleres_dataset(root_dir=args.data_path,generate_latents = True)
    tr_dataset = Res_128_dataset(root_dir=root_dir,files_names_path=files_names_path,generate_latents = True)
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
    
    for sample,names in tr_dataloader:
        sample = sample.cuda()
        with torch.no_grad():
            #z =  AE.patch_encode(sample,patch_size = 64)
            z = AE.encode(sample,quantize=False)
            #pdb.set_trace()
            output = ((z - AE.codebook.embeddings.min()) /
            (AE.codebook.embeddings.max() -
            AE.codebook.embeddings.min())) * 2.0 - 1.0
        output = output.cpu()
        for idx, path in enumerate(names):
            #pdb.set_trace()
            output_ = output[idx]
            out_put_path = os.path.join(latent_ds_save_root, f"latent_{names[idx]}.nii.gz")
            img = tio.ScalarImage(tensor = output_ )
            img.save(out_put_path)   

if __name__ == "__main__":
    generate()



