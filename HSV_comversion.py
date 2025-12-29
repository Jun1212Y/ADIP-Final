import cv2
import matplotlib.pyplot as plt

# 1. Load the image
img_bgr = cv2.imread('result.jpg')

# 2. Convert from BGR to HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# 3. Split the channels for visualization
h, s, v = cv2.split(img_hsv)

# (Optional) Display using Matplotlib
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Original (Convert to RGB for correct display)
axs[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original (RGB)')

# Hue (Color Spectrum)
axs[1].imshow(h, cmap='jet') # 'jet' colormap helps visualize rotation of color
axs[1].set_title('Hue Channel (Color)')

# Saturation (Intensity)
axs[2].imshow(s, cmap='gray')
axs[2].set_title('Saturation Channel (Vibrancy)')

# Value (Brightness)
axs[3].imshow(v, cmap='gray')
axs[3].set_title('Value Channel (Brightness)')

for ax in axs: ax.axis('off')
plt.show()
