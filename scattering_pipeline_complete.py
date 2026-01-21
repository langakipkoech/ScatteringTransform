import torch
from kymatio.torch import Scattering2D
import cv2
from matplotlib import pyplot as plt
import numpy as np

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



path = ['ocean_blender.png']

img = None
img_path = None
for pat in path:
    if os.path.exists(pat):
        img = cv2.imread(pat, 0)
        if img is not None:
            img_path = pat
            print(f"Image loaded from: {pat}")
            break

if img is None:
    raise FileNotFoundError(
        "Could not find path. Please update the path in the script.\n"
       )

print(f"Image shape: {img.shape}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

#scattering initialization
scattering = Scattering2D(J=2, shape=img.shape).to(device)

# Convert image to PyTorch tensor 
img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)


# Calculate scattering transform 
with torch.no_grad():
    Sx = scattering(img_tensor)

print(f"Scattering coefficients shape: {Sx.shape}")
print(f"Scattering coefficients on: {Sx.device}")

# Scattering coefficients
Sx_cpu = Sx.cpu().numpy()

Sx_squeezed = Sx_cpu[0, 0]  

plt.subplot(1, 2, 2)
plt.imshow(Sx_squeezed[0], cmap='viridis')
plt.title('Zeroth Order Coefficient (Lowpass)', fontsize=14, weight='bold')
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.show()

# Visualize individual scattering coefficients
n_coeffs = min(16, Sx_squeezed.shape[0])
n_rows = 4
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
axes = axes.flatten()

for i in range(n_coeffs):
    im = axes[i].imshow(Sx_squeezed[i], cmap='viridis')
    axes[i].set_title(f'Coefficient {i}', fontsize=14, weight='bold')
    axes[i].axis('off')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)


for i in range(n_coeffs, len(axes)):
    axes[i].axis('off')

plt.suptitle(f'Scattering Transform Coefficients', 
             fontsize=14, y=0.995, weight='bold')
plt.tight_layout()
plt.show()


del Sx, img_tensor
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA cache cleared")
    
    