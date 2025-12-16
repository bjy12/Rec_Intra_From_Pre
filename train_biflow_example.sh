#!/bin/bash
# BiFlowNet 训练示例脚本（Linux/Mac）
# 请根据您的实际路径修改以下参数

python train/train_BiFlowNet_SingleRes_single_gpu.py \
    --data-path "D:/data_space/Zhongrifriendly/paired_process_128_tigre/latent_ds/" \
    --files-names-path "./files_names/train_files.txt" \
    --AE-ckpt "D:/code_space_bone/3D-MedDiffusion-main/ver_128_full_VQAE/results/my_model/version_0/checkpoints/latest_checkpoint-v2.ckpt" \
    --results-dir "./results_biflow/" \
    --num-classes 1 \
    --batch-size 4 \
    --epochs 1000 \
    --resolution 32 32 32 \
    --volume-channels 8 \
    --model-dim 72 \
    --dim-mults 1 1 2 4 8 \
    --use-attn 0 0 0 1 1 \
    --patch-size 1 \
    --num-dit 1 \
    --vq-size 64 \
    --timesteps 1000 \
    --loss-type l1 \
    --log-every 50 \
    --ckpt-every 500 \
    --num-workers 4 \
    --global-seed 0 \
    --enable_amp



