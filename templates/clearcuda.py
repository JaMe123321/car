import torch

# 清空 PyTorch 的 CUDA 緩存
torch.cuda.empty_cache()

# 強制垃圾回收
import gc
gc.collect()

print("CUDA 暫存器已清空")
