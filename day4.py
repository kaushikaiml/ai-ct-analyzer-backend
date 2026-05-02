import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from scipy import ndimage

# Image load karo
image = Image.open(r"C:\Users\mohit\ai-ct-analyzer\data\ct_scan.png")
img_array = np.array(image).astype(np.float32)

# Sirf lung area lo
lung = img_array[80:180, 80:230]

# Small bright spots dhundho
bright = (lung > 100) & (lung < 160)

# Connected regions label karo
labeled, num = ndimage.label(bright)
print(f"Total suspicious spots: {num}")

# Image dikhao boxes ke saath
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(lung, cmap="gray")

# Har region pe box draw karo
for i in range(1, num + 1):
    region = np.where(labeled == i)
    if len(region[0]) > 5:
        rmin, rmax = region[0].min(), region[0].max()
        cmin, cmax = region[1].min(), region[1].max()
        box = patches.Rectangle(
            (cmin, rmin), cmax-cmin, rmax-rmin,
            linewidth=1, edgecolor="red", facecolor="none"
        )
        ax.add_patch(box)

ax.set_title("Suspicious Nodules Detected!")
plt.show()
print("Day 4 Complete!")