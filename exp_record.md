## 1. 训练vq ae 压缩ct  
   python ./train/train_AE_full_ct_128.py --config ./config/PatchVolume_128x128x128.yaml
## 2. 制作latent feature  map 数据集
   python ./train/generate_training_latent.py 
## 3. 训练biflow 
   python .\train\train_BiFlowNet_Rec.py --config .\config\BiFlow_Rec_PreCT_Intra.yaml
第一次训练效果不行，完全没有效果。validation 在step 11000 之前可以看到可以看到钉子被重建的结果，step 11000 之后变为了网格。 有完全不能得到正常的结果，从训练集可以稍微能够看的出一点骨骼结构。
   网络是否在step 115000 之后发生了训练的崩溃。原因: 采用了batch_size=2, 但是xray encoder 内部使用了batch norm的方式来归一化，较低的bath_size 并不适合这种归一化的方式。 要采用梯度积累? 还是修改位group norm ? 来避免这种情况发生? 
   #* 


    