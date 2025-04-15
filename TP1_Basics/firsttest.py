import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import numpy as np
import os

# Load image
image = imread('../images/retina.jpg')  # Updated path
print(type(image))
print(image.shape, image.dtype)

# Display full image
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Display RGB channels
red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(red, cmap='gray')
plt.title("Red Channel")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(green, cmap='gray')
plt.title("Green Channel")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(blue, cmap='gray')
plt.title("Blue Channel")
plt.axis("off")
plt.show()

# Save image with different JPEG quality
imsave("../images/compressed_q10.jpg", image, quality=10)
imsave("../images/compressed_q50.jpg", image, quality=50)
imsave("../images/compressed_q90.jpg", image, quality=90)