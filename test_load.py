import matplotlib.pyplot as plt
from PIL import Image

# CT scan image load karo
image = Image.open(r"C:\Users\mohit\ai-ct-analyzer\data\ct_scan.png")
# Screen pe dikhao
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap="gray")
plt.title("Meri Pehli CT Scan Image!")
plt.axis("off")
plt.show()

print("Day 1 Complete!")