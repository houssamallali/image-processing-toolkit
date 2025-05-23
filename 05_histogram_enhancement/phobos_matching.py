import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.exposure import equalize_hist
import sys

def custom_histogram_equalization(image):
    """
    Apply histogram equalization to an image
    T(x_k) = L * cdf_I(k)
    """
    # Get image histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 1))
    
    # Calculate cumulative distribution function
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]  # Normalize to [0,1]
    
    # Apply histogram equalization using CDF as a mapping function
    equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    
    return equalized.reshape(image.shape)

def custom_histogram_matching(source, target_cdf, target_bins):
    """
    Transform source image to match a target histogram
    x₂ = cdf₂⁻¹(cdf₁(x₁))
    
    Parameters:
    -----------
    source : ndarray
        Source image to be transformed
    target_cdf : ndarray
        Target cumulative distribution function
    target_bins : ndarray
        Target intensity bins
        
    Returns:
    --------
    matched : ndarray
        The transformed source image
    """
    # Get source histogram and CDF
    src_hist, src_bins = np.histogram(source.flatten(), bins=256, range=(0, 1))
    src_cdf = np.cumsum(src_hist)
    src_cdf_normalized = src_cdf / src_cdf[-1]  # Normalize to [0,1]
    
    # Step 1: Map source pixel values to their corresponding CDF values
    src_cdf_values = np.interp(source.flatten(), src_bins[:-1], src_cdf_normalized)
    
    # Step 2: Find target intensity values with the same CDF value
    matched = np.interp(src_cdf_values, target_cdf, target_bins[:-1])
    
    return matched.reshape(source.shape)

# Load the Phobos image (with fallback to a synthetic image)
try:
    phobos = img_as_float(imread('../images/phobos.png', as_gray=True))
    image_name = 'Phobos'
except FileNotFoundError:
    # Create a synthetic image as fallback
    print("Phobos image not found, creating synthetic image")
    x, y = np.meshgrid(np.linspace(-5, 5, 256), np.linspace(-5, 5, 256))
    phobos = np.sin(x**2 + y**2) / (x**2 + y**2 + 1)
    phobos = (phobos - phobos.min()) / (phobos.max() - phobos.min())  # Normalize to [0,1]
    image_name = 'Synthetic'

# Create a target histogram (similar to the one in Figure 2.6f)
# This creates a distribution with a prominent peak at low intensity
# and a smaller peak at higher intensity
x = np.linspace(0, 1, 256)
target_hist = np.exp(-50 * (x - 0.1)**2) + 0.2 * np.exp(-30 * (x - 0.6)**2)
target_hist = target_hist / np.sum(target_hist)

# Calculate target CDF
target_cdf = np.cumsum(target_hist)
target_bins = np.linspace(0, 1, 256+1)

# Apply histogram equalization
equalized = custom_histogram_equalization(phobos)

# Apply histogram matching with our bimodal target
matched = custom_histogram_matching(phobos, target_cdf, target_bins)

# Create a figure layout similar to Figure 2.6
plt.figure(figsize=(15, 12))

# Row 1: Images
plt.subplot(331)
plt.imshow(phobos, cmap='gray')
plt.title('(a) Original image')
plt.axis('off')

plt.subplot(332)
plt.imshow(equalized.reshape(phobos.shape), cmap='gray')
plt.title('(b) Histogram equalization')
plt.axis('off')

plt.subplot(333)
plt.imshow(matched.reshape(phobos.shape), cmap='gray')
plt.title('(c) Histogram matching')
plt.axis('off')

# Row 2: Histograms of original and equalized
plt.subplot(334)
hist_orig, bins_orig = np.histogram(phobos.flatten(), bins=256, range=(0, 1))
plt.bar(bins_orig[:-1], hist_orig, width=1/256, color='black')
plt.title('(d) Histogram of original image')
plt.xlim(0, 1)
plt.grid(alpha=0.3)

plt.subplot(335)
hist_eq, bins_eq = np.histogram(equalized, bins=256, range=(0, 1))
plt.bar(bins_eq[:-1], hist_eq, width=1/256, color='black')
plt.title('(e) Histogram after equalization')
plt.xlim(0, 1)
plt.grid(alpha=0.3)

# Row 3: Target histogram and result after matching
plt.subplot(336)
hist_matched, bins_matched = np.histogram(matched, bins=256, range=(0, 1))
plt.bar(bins_matched[:-1], hist_matched, width=1/256, color='black')
plt.title('(g) Histogram after matching')
plt.xlim(0, 1)
plt.grid(alpha=0.3)

plt.subplot(337)
plt.plot(x, target_hist, 'b-')
plt.title('(f) Target histogram')
plt.xlim(0, 1)
plt.grid(alpha=0.3)

# Plot the CDFs
plt.subplot(338)
src_hist, src_bins = np.histogram(phobos.flatten(), bins=256, range=(0, 1))
src_cdf = np.cumsum(src_hist) / np.sum(src_hist)
plt.plot(src_bins[:-1], src_cdf, 'b-', label='Source CDF')
plt.plot(target_bins[:-1], target_cdf, 'r-', label='Target CDF')
plt.title('Cumulative Distribution Functions')
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)

# Extract a small region to demonstrate the mapping
plt.subplot(339)
# Create a simplified version of Figure 2.2 to demonstrate the principle
plt.plot(src_bins[:-1], src_cdf, 'b-', label='cdf₁(x₁)')
plt.plot(target_bins[:-1], target_cdf, 'r-', label='cdf₂(x₂)')

# Add arrows to demonstrate the mapping for a specific value
x1 = 0.3
y = np.interp(x1, src_bins[:-1], src_cdf)
x2 = np.interp(y, target_cdf, target_bins[:-1])

plt.plot([x1, x1], [0, y], 'k--')
plt.plot([x1, x2], [y, y], 'k--')
plt.plot([x2, x2], [y, 0], 'k--')

plt.scatter(x1, y, color='black', s=30)
plt.scatter(x2, y, color='black', s=30)

plt.text(x1-0.05, -0.07, 'x₁', fontsize=12)
plt.text(x2-0.05, -0.07, 'x₂', fontsize=12)

plt.title('Histogram matching principle')
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.suptitle(f'Histogram Matching of {image_name} Image', fontsize=16, y=0.99)
plt.subplots_adjust(top=0.95)
plt.show() 