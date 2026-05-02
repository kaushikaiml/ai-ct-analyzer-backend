import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image = Image.open(r"C:\Users\mohit\ai-ct-analyzer\data\ct_scan.png")
img_array = np.array(image)

# Sirf lung area zoom karke dikhao
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(img_array, cmap="gray")
axes[0].set_title("Poora Scan")

# Lung area zoom
axes[1].imshow(img_array[50:200, 50:250], cmap="gray")
axes[1].set_title("Lungs Zoom")

plt.tight_layout()
plt.show()

print("Pixel count:", img_array.shape[0] * img_array.shape[1])
