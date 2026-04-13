import numpy as np
import matplotlib.pyplot as plt

data = np.load("../samples/samples-2026-04-13-13-50-37/samples_10x64x64x3.npz")
images = data['arr_0']

num_images = min(16, len(images)) # 最多看16张
rows = 4
cols = 4

fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
for i in range(num_images):
    ax = axes[i // cols, i % cols]
    ax.imshow(images[i])
    ax.axis('off')

plt.tight_layout()
plt.show()