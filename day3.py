import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Image load karo
image = Image.open(r"C:\Users\mohit\ai-ct-analyzer\data\ct_scan.png")
img_array = np.array(image).astype(np.float32)

# Bright spots dhundho
bright_areas = (img_array > 200).astype(np.float32)

# Result dikhao
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_array, cmap="gray")
axes[0].set_title("Original CT Scan")

axes[1].imshow(bright_areas, cmap="gray")
axes[1].set_title("Bright Areas Detected")

axes[2].imshow(img_array, cmap="gray")
axes[2].imshow(bright_areas, cmap="Reds", alpha=0.4)
axes[2].set_title("Suspicious Areas Highlighted")

plt.tight_layout()
plt.show()

print("Total bright pixels:", int(bright_areas.sum()))
print("Day 3 Complete!")