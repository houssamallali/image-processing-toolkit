import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.exposure import adjust_gamma

# Load the osteoblast image
image = img_as_float(imread('../images/osteoblaste.jpg', as_gray=True))

# Apply gamma correction with different values
gamma_values = [0.001, 1.0, 4]
corrected_images = [adjust_gamma(image, gamma=g) for g in gamma_values]

# Plot the results
plt.figure(figsize=(12, 8))

# Original image
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Gamma corrected images
for i, (img, gamma) in enumerate(zip(corrected_images, gamma_values)):
    plt.subplot(2, 2, i+2)
    plt.imshow(img, cmap='gray')
    plt.title(f'Gamma = {gamma}')
    plt.axis('off')

plt.tight_layout()
plt.show() 