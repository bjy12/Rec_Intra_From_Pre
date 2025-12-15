## 1. 训练vq ae 压缩ct  
   python ./train/train_AE_full_ct_128.py --config ./config/PatchVolume_128x128x128.yaml
## 2. 制作latent feature  map 数据集
   python ./train/generate_training_latent.py 
## 3. 训练biflow 
   
    