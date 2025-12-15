# BiFlowNet 训练完整指南

## 📋 目录
1. [训练流程概述](#训练流程概述)
2. [训练前准备](#训练前准备)
3. [脚本结构解析](#脚本结构解析)
4. [训练步骤](#训练步骤)
5. [参数说明](#参数说明)
6. [常见问题](#常见问题)

---

## 🎯 训练流程概述

BiFlowNet 训练是 3D MedDiffusion 项目的第二阶段，需要先完成 AutoEncoder 训练。

### 完整训练流程

```
步骤 1: 训练 AutoEncoder (Stage 1)
  ↓
步骤 2: 训练 AutoEncoder (Stage 2)  
  ↓
步骤 3: 编码图像到潜在空间
  ↓
步骤 4: 训练 BiFlowNet 扩散模型
  ↓
完成：可以生成 3D 医学图像
```

---

## 📦 训练前准备

### 1. 完成 AutoEncoder 训练

确保你已经完成了：
- ✅ Stage 1 训练：`train_PatchVolume.py`
- ✅ Stage 2 训练：`train_PatchVolume_stage2.py`
- ✅ 获得 Stage 2 检查点：`PatchVolume_8x_s2.ckpt` 或 `PatchVolume_4x_s2.ckpt`

### 2. 准备数据配置文件

创建 `config/Singleres_dataset.json`：

```json
{
    "0": "path/to/CTHeadNeck/images",
    "1": "path/to/CTChestAbdomen/images",
    "2": "path/to/CTLegs/images",
    "3": "path/to/MRT1Brain/images",
    "4": "path/to/MRT2Brain/images",
    "5": "path/to/MRAbdomen/images",
    "6": "path/to/MRKnee/images"
}
```

**说明**：
- 键（"0", "1", ...）：类别索引，对应不同的解剖区域/模态
- 值：图像文件所在的目录路径

### 3. 编码图像到潜在空间

**重要**：BiFlowNet 训练需要潜在空间数据，不是原始图像！

```bash
python train/generate_training_latent.py \
    --data-path config/Singleres_dataset.json \
    --AE-ckpt checkpoints/PatchVolume_8x_s2.ckpt \
    --batch-size 4 \
    --num-workers 8
```

**这个过程会**：
- 读取原始图像（`.nii.gz`）
- 使用 AutoEncoder 编码到潜在空间
- 保存潜在表示为 `.nii.gz` 文件到 `{原目录}_latents/`

**输出结构**：
```
path/to/CTHeadNeck/images/          # 原始图像
path/to/CTHeadNeck/images_latents/  # 潜在表示（训练 BiFlowNet 使用）
```

---

## 🔍 脚本结构解析

### `train_BiFlowNet_SingleRes.py` 主要组成部分

#### 1. **初始化阶段**（第123-216行）

```python
# 1.1 分布式训练设置（DDP）
dist.init_process_group("nccl")  # 多 GPU 训练（需要 NCCL，仅 Linux）

# 1.2 创建 BiFlowNet 模型
model = BiFlowNet(
    dim=72,                    # 模型维度
    cond_classes=7,            # 类别数量
    channels=8,                 # 潜在空间通道数
    timesteps=1000,            # 扩散步数
    ...
)

# 1.3 创建扩散过程
diffusion = GaussianDiffusion(
    channels=8,
    timesteps=1000,
    loss_type='l1'
)

# 1.4 加载预训练的 AutoEncoder（冻结）
AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt)
AE.eval()  # 只用于编码/解码，不训练

# 1.5 创建 EMA 模型（指数移动平均）
ema_model = copy.deepcopy(model)
```

#### 2. **数据加载**（第218-224行）

```python
# 加载潜在空间数据
dataset = Singleres_dataset(args.data_path, resolution=args.resolution)
# 返回：(latent, class_idx, resolution)
# - latent: (8, 32, 32, 32) 潜在表示
# - class_idx: 0-6 类别索引
# - resolution: (32,32,32)/64.0 分辨率嵌入
```

#### 3. **训练循环**（第241-327行）

```python
for epoch in range(start_epoch, args.epochs):
    for z, y, res in loader:  # z: 潜在表示, y: 类别, res: 分辨率
        # 3.1 随机采样时间步
        t = torch.randint(0, 1000, (b,))
        
        # 3.2 计算扩散损失
        loss = diffusion.p_losses(model, z, t, y=y, res=res)
        # 内部过程：
        #   - 添加噪声：z_t = sqrt(alpha_t) * z + sqrt(1-alpha_t) * noise
        #   - 预测噪声：noise_pred = BiFlowNet(z_t, t, y, res)
        #   - 计算损失：loss = L1(noise, noise_pred)
        
        # 3.3 反向传播
        loss.backward()
        opt.step()
        
        # 3.4 更新 EMA（每 10 步）
        if train_steps % 10 == 0:
            ema.update_model_average(ema_model, model)
```

#### 4. **检查点保存和样本生成**（第286-326行）

每 `ckpt_every` 步（默认 500 步）：
- 保存检查点（模型、EMA、优化器状态）
- 生成样本图像（监控训练进度）

---

## 🚀 训练步骤

### 方法 1：多 GPU 训练（Linux）

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port 29513 \
    train/train_BiFlowNet_SingleRes.py \
    --data-path config/Singleres_dataset.json \
    --results-dir ./results/biflow \
    --AE-ckpt checkpoints/PatchVolume_8x_s2.ckpt \
    --num-classes 7 \
    --resolution 32 32 32 \
    --batch-size 48 \
    --num-workers 48 \
    --epochs 1000 \
    --ckpt-every 500 \
    --log-every 50
```

### 方法 2：单 GPU 训练（Windows/Linux）

由于原脚本使用 DDP（需要 NCCL），Windows 不支持。需要使用单 GPU 版本（见下方）。

---

## 📝 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--data-path` | 数据配置文件路径 | `config/Singleres_dataset.json` |
| `--results-dir` | 结果保存目录 | `./results/biflow` |
| `--AE-ckpt` | AutoEncoder 检查点路径 | `checkpoints/PatchVolume_8x_s2.ckpt` |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-classes` | 7 | 类别数量（对应数据配置中的键） |
| `--resolution` | `32 32 32` | 潜在空间分辨率 |
| `--volume-channels` | 8 | 潜在空间通道数（必须与 AE 匹配） |
| `--model-dim` | 72 | 模型维度 |
| `--dim-mults` | `1 1 2 4 8` | U-Net 通道倍数 |
| `--use-attn` | `0 0 0 1 1` | 各层是否使用注意力 |
| `--patch-size` | 1 | Patch 大小 |
| `--num-dit` | 1 | DiT 块数量 |
| `--vq-size` | 64 | VQ 大小 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 16 | Batch 大小 |
| `--epochs` | 1000 | 训练轮数 |
| `--num-workers` | 8 | 数据加载线程数 |
| `--log-every` | 50 | 每 N 步打印日志 |
| `--ckpt-every` | 500 | 每 N 步保存检查点 |
| `--ckpt` | `None` | 恢复训练的检查点路径 |
| `--loss-type` | `l1` | 损失类型（l1 或 l2） |
| `--timesteps` | 1000 | 扩散步数 |
| `--enable_amp` | `True` | 混合精度训练 |

---

## ⚠️ Windows 用户注意

原脚本使用 `torch.distributed`（DDP），需要 NCCL，**Windows 不支持**。

**解决方案**：使用单 GPU 版本（见下方创建的脚本）

---

## 📊 训练输出

训练过程中会生成：

```
results/biflow/
└── 000-BiFlowNet/
    ├── checkpoints/
    │   ├── 0000500.pt  # 每 500 步保存
    │   ├── 0001000.pt
    │   └── ...
    ├── samples/
    │   ├── 1_0.nii.gz  # 生成的样本（类别 0）
    │   ├── 1_1.nii.gz  # 生成的样本（类别 1）
    │   └── ...
    └── log.txt  # 训练日志
```

---

## 🔧 常见问题

### Q1: 数据格式错误
**错误**：`FileNotFoundError: No .nii.gz files found`

**解决**：
1. 确保已经运行 `generate_training_latent.py` 生成潜在表示
2. 检查数据配置文件中的路径是否正确
3. 确保路径指向 `*_latents` 目录

### Q2: 显存不足
**解决**：
- 减小 `--batch-size`（例如：48 → 16 → 8）
- 减小 `--resolution`（例如：32 32 32 → 16 16 16）
- 使用梯度累积（需要修改代码）

### Q3: DDP 错误（Windows）
**错误**：`RuntimeError: Distributed package doesn't have NCCL built in`

**解决**：使用单 GPU 版本脚本（见下方）

---

## 📈 训练监控

### 查看训练日志

```bash
# 实时查看日志
tail -f results/biflow/000-BiFlowNet/log.txt
```

### 检查生成的样本

```bash
# 查看 samples 目录
ls results/biflow/000-BiFlowNet/samples/
```

---

## 🎓 理解训练过程

### 扩散模型训练原理

1. **前向过程**（添加噪声）：
   ```
   z_0 (干净潜在表示)
     ↓ 添加噪声
   z_t (带噪声的潜在表示)
   ```

2. **反向过程**（去噪）：
   ```
   z_t (带噪声)
     ↓ BiFlowNet 预测噪声
   noise_pred
     ↓ 减去预测噪声
   z_{t-1} (更干净)
     ↓ 重复 1000 次
   z_0 (生成的潜在表示)
   ```

3. **训练目标**：
   - 学习预测噪声：`noise_pred = BiFlowNet(z_t, t, y, res)`
   - 最小化：`loss = L1(真实噪声, 预测噪声)`

### EMA 的作用

- **EMA（指数移动平均）**：平滑模型权重
- 公式：`θ_ema = 0.995 * θ_ema + 0.005 * θ_current`
- 作用：生成更稳定的样本
- 使用：推理时使用 `ema_model` 而不是 `model`

---

## ✅ 训练检查清单

- [ ] 完成 AutoEncoder Stage 1 和 Stage 2 训练
- [ ] 准备数据配置文件 `Singleres_dataset.json`
- [ ] 运行 `generate_training_latent.py` 生成潜在表示
- [ ] 检查潜在表示文件是否存在
- [ ] 准备足够的 GPU 显存（建议 24GB+）
- [ ] 设置合适的训练参数
- [ ] 开始训练并监控日志

---

## 📚 下一步

训练完成后，可以使用训练好的模型进行推理：

```bash
python evaluation/class_conditional_generation.py \
    --AE-ckpt checkpoints/PatchVolume_8x_s2.ckpt \
    --model-ckpt results/biflow/000-BiFlowNet/checkpoints/0005000.pt \
    --output-dir ./generated_samples
```

