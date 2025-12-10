import torch
from PIL import Image
import numpy as np

img_path = r"C:\Users\natha\OneDrive\Documents\Desktop\choom\choom\output\b6f2_4023.png"

img = Image.open(img_path).convert("L")
img_array = np.array(img)  
print("Numpy array shape:", img_array.shape)
tensor = torch.from_numpy(img_array)  
print("Tensor shape:", tensor.shape)
print("Tensor dtype:", tensor.dtype)

#float32 and normalize
tensor = tensor.float() / 255.0  # range [0,1]

tensor = tensor.unsqueeze(0)  # shape: (1, H, W)
print("Tensor shape after adding channel:", tensor.shape)
