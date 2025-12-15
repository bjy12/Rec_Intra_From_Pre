"""
快速检查 CUDA 是否可用
"""
import torch

print("=" * 50)
print("CUDA 检查")
print("=" * 50)

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("\n❌ CUDA 不可用！")
    print("\n可能的原因：")
    print("1. 没有安装支持 CUDA 的 PyTorch")
    print("2. 没有 NVIDIA GPU")
    print("3. GPU 驱动未安装或版本不匹配")
    print("\n解决方案：")
    print("如果安装了 PyTorch 2.6.0，请检查是否安装了 CUDA 版本：")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("=" * 50)

