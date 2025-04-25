'''
Verify whether cuda are avaliable for training
'''


import torch
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be True
print(f"GPU count: {torch.cuda.device_count()}")       # Should be ≥1
print(f"Current GPU: {torch.cuda.get_device_name(0)}") # Should show RTX 3050