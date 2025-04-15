import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.exposure import match_histograms

def custom_hist_match(source, target):
    """
    Adjust the pixel values of source to match the histogram of target
    
    Parameters:
    -----------
    source : ndarray
        Image to transform
    target : ndarray
        Reference image
        
    Returns:
    --------
    matched : ndarray
        The transformed source image
    """
    # Compute histograms and CDFs
    src_hist, src_bins = np.histogram(source.flatten(), bins=256, range=(0, 1), density=True)
    tgt_hist, tgt_bins = np.histogram(target.flatten(), bins=256, range=(0, 1), density=True)
    
    # Calculate CDFs
    src_cdf = np.cumsum(src_hist)
    src_cdf = src_cdf / src_cdf[-1]  # Normalize
    
    tgt_cdf = np.cumsum(tgt_hist)
    tgt_cdf = tgt_cdf / tgt_cdf[-1]  # Normalize
    
    # Step 1: Map source pixel to its CDF value
    src_mapped = np.interp(source.flatten(), src_bins[:-1], src_cdf)
    
    # Step 2: Find the corresponding intensity in target using inverse CDF
    # Create a mapping from CDF values back to pixel intensities in target
    inv_map = np.zeros_like(src_cdf)
    for i, v in enumerate(tgt_cdf):
        # Find where each cdf value would be in the source cdf
        idx = np.searchsorted(src_cdf, v)
        if idx >= len(src_cdf):
            idx = len(src_cdf) - 1
        inv_map[i] = src_bins[idx]
        
    # Apply the mapping to get matched image
    matched = np.interp(src_mapped, tgt_cdf, tgt_bins[:-1])
    
    return matched.reshape(source.shape)

# Load images
try:
    # Try to load Phobos image as source
    source = img_as_float(imread('../images/phobos.png', as_gray=True))
    source_name = 'Phobos'
except:
    # Fallback to osteoblast
    source = img_as_float(imread('../images/osteoblaste.jpg', as_gray=True))
    source_name = 'Osteoblast'

# Create a bimodal target histogram (synthetic image with bimodal distribution)
x = np.linspace(0, 1, 256)
# Bimodal distribution with peaks at 0.3 and 0.7
bimodal_hist = np.exp(-50 * (x - 0.3)**2) + 0.7 * np.exp(-50 * (x - 0.7)**2)
bimodal_hist = bimodal_hist / np.sum(bimodal_hist)  # Normalize

# Create a synthetic target image with this histogram
target = np.random.choice(x, size=source.shape, p=bimodal_hist)

# Apply histogram matching
custom_matched = custom_hist_match(source, target)
skimage_matched = match_histograms(source, target)

# Plot results
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Source image and histogram
axes[0, 0].imshow(source, cmap='gray')
axes[0, 0].set_title(f'Source: {source_name}')
axes[0, 0].axis('off')

axes[0, 1].hist(source.flatten(), bins=256, range=(0, 1), color='black')
axes[0, 1].set_title('Source Histogram')
axes[0, 1].set_xlim([0, 1])

# Target histogram
axes[1, 0].imshow(target, cmap='gray')
axes[1, 0].set_title('Target (Bimodal)')
axes[1, 0].axis('off')

axes[1, 1].hist(target.flatten(), bins=256, range=(0, 1), color='black')
axes[1, 1].set_title('Target Histogram (Bimodal)')
axes[1, 1].set_xlim([0, 1])

# Custom matched
axes[2, 0].imshow(custom_matched, cmap='gray')
axes[2, 0].set_title('Custom Histogram Matching')
axes[2, 0].axis('off')

axes[2, 1].hist(custom_matched.flatten(), bins=256, range=(0, 1), color='black')
axes[2, 1].set_title('Custom Matched Histogram')
axes[2, 1].set_xlim([0, 1])

# Scikit-image matched
axes[0, 2].imshow(skimage_matched, cmap='gray')
axes[0, 2].set_title('Scikit-image Matched')
axes[0, 2].axis('off')

axes[1, 2].hist(skimage_matched.flatten(), bins=256, range=(0, 1), color='black')
axes[1, 2].set_title('Scikit-image Matched Histogram')
axes[1, 2].set_xlim([0, 1])

# Plot the matching CDFs
axes[2, 2].plot(x, np.cumsum(bimodal_hist), 'r-', label='Target CDF')
axes[2, 2].set_title('Cumulative Distribution Functions')
axes[2, 2].set_xlabel('Intensity')
axes[2, 2].set_ylabel('CDF')
axes[2, 2].legend()

plt.tight_layout()
plt.show() 