import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.exposure import equalize_hist

def display_hist(image, title='Histogram'):
    """Display histogram of an image"""
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 1))
    plt.plot(bins[:-1], hist, 'k-')
    plt.title(title)
    plt.xlim([0, 1])
    plt.tight_layout()

def custom_hist_eq(image):
    """
    Custom implementation of histogram equalization
    
    T(x_k) = L * cdf_I(k)
    """
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 1))
    
    # Calculate CDF
    cdf = np.cumsum(hist)
    
    # Normalize CDF to [0, 1]
    cdf = cdf / cdf[-1]
    
    # Map pixel values using the CDF
    image_eq = np.interp(image.flatten(), bins[:-1], cdf)
    
    # Reshape to original image dimensions
    return image_eq.reshape(image.shape)

# Load the image (osteoblast)
image = img_as_float(imread('../images/osteoblaste.jpg', as_gray=True))

# Apply histogram equalization
custom_eq = custom_hist_eq(image)
skimage_eq = equalize_hist(image)

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original image and histogram
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].hist(image.flatten(), bins=256, range=(0, 1), color='black')
axes[0, 1].set_title('Original Histogram')
axes[0, 1].set_xlim([0, 1])

# Custom equalization
axes[1, 0].imshow(custom_eq, cmap='gray')
axes[1, 0].set_title('Custom Equalization')
axes[1, 0].axis('off')

axes[1, 1].hist(custom_eq.flatten(), bins=256, range=(0, 1), color='black')
axes[1, 1].set_title('Custom Equalized Histogram')
axes[1, 1].set_xlim([0, 1])

# Scikit-image equalization
axes[0, 2].imshow(skimage_eq, cmap='gray')
axes[0, 2].set_title('Scikit-image Equalization')
axes[0, 2].axis('off')

axes[1, 2].hist(skimage_eq.flatten(), bins=256, range=(0, 1), color='black')
axes[1, 2].set_title('Scikit-image Equalized Histogram')
axes[1, 2].set_xlim([0, 1])

plt.tight_layout()
plt.show() 