import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float, img_as_ubyte
from skimage.exposure import equalize_hist, equalize_adapthist

# Load the osteoblast image
image = img_as_float(imread('../images/osteoblaste.jpg', as_gray=True))

# Apply histogram equalization
hist_eq = equalize_hist(image)

# Apply adaptive histogram equalization (CLAHE)
adapthist_eq = equalize_adapthist(image, clip_limit=0.03)

# Calculate histograms
bins = 256
hist_original = np.histogram(image, bins=bins, range=(0, 1))[0]
hist_equalized = np.histogram(hist_eq, bins=bins, range=(0, 1))[0]
hist_adapthist = np.histogram(adapthist_eq, bins=bins, range=(0, 1))[0]

# Plot the results
plt.figure(figsize=(15, 10))

# Original image and histogram
plt.subplot(3, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.plot(hist_original, color='black')
plt.title('Original Histogram')
plt.xlim([0, bins-1])
plt.tight_layout()

# Histogram equalized image and histogram
plt.subplot(3, 2, 3)
plt.imshow(hist_eq, cmap='gray')
plt.title('Histogram Equalized')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.plot(hist_equalized, color='black')
plt.title('Equalized Histogram')
plt.xlim([0, bins-1])
plt.tight_layout()

# Adaptive histogram equalized image and histogram
plt.subplot(3, 2, 5)
plt.imshow(adapthist_eq, cmap='gray')
plt.title('Adaptive Histogram Equalization (CLAHE)')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.plot(hist_adapthist, color='black')
plt.title('CLAHE Histogram')
plt.xlim([0, bins-1])
plt.tight_layout()

plt.subplots_adjust(hspace=0.3)
plt.show() 