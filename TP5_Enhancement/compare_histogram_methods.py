import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.exposure import equalize_hist, equalize_adapthist, match_histograms
from skimage.color import rgb2gray
import sys

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

def plot_img_with_hist(img, title, ax_img, ax_hist):
    """Plot an image and its histogram on the given axes"""
    ax_img.imshow(img, cmap='gray')
    ax_img.set_title(title)
    ax_img.axis('off')
    
    ax_hist.hist(img.flatten(), bins=256, range=(0, 1), color='black')
    ax_hist.set_title(f'Histogram: {title}')
    ax_hist.set_xlim([0, 1])

# Try to load both osteoblast and phobos images
try:
    osteoblast = img_as_float(imread('../images/osteoblaste.jpg', as_gray=True))
    phobos = img_as_float(imread('../images/phobos.png', as_gray=True))
    # We have both images
    images = {
        'Osteoblast': osteoblast,
        'Phobos': phobos
    }
except:
    # Fallback to using just one image
    try:
        img = img_as_float(imread('../images/osteoblaste.jpg', as_gray=True))
        images = {'Osteoblast': img}
    except:
        try:
            img = img_as_float(imread('../images/phobos.png', as_gray=True))
            images = {'Phobos': img}
        except:
            # Backup option - create a synthetic image
            x, y = np.meshgrid(np.linspace(-5, 5, 256), np.linspace(-5, 5, 256))
            img = np.sin(x**2 + y**2) / (x**2 + y**2 + 1)
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
            images = {'Synthetic': img}

# Create a bimodal target histogram (for histogram matching)
x = np.linspace(0, 1, 256)
bimodal_hist = np.exp(-50 * (x - 0.3)**2) + 0.7 * np.exp(-50 * (x - 0.7)**2)
bimodal_hist = bimodal_hist / np.sum(bimodal_hist)
bimodal_target = np.random.choice(x, size=(256, 256), p=bimodal_hist)

# Plot results for each image
for name, img in images.items():
    # Apply enhancement methods
    eq_hist = equalize_hist(img)
    adapt_hist = equalize_adapthist(img, clip_limit=0.03)
    hist_match = match_histograms(img, bimodal_target)
    
    # Set up the plot grid
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    fig.suptitle(f'Histogram Enhancement Comparison: {name}', fontsize=16)
    
    # Original image
    plot_img_with_hist(img, 'Original', axes[0, 0], axes[0, 1])
    
    # Histogram equalization
    plot_img_with_hist(eq_hist, 'Histogram Equalization', axes[1, 0], axes[1, 1])
    
    # Adaptive histogram equalization
    plot_img_with_hist(adapt_hist, 'Adaptive Histogram Eq. (CLAHE)', axes[2, 0], axes[2, 1])
    
    # Histogram matching
    plot_img_with_hist(hist_match, 'Histogram Matching (Bimodal)', axes[3, 0], axes[3, 1])
    
    # Target histogram (for reference)
    plot_img_with_hist(bimodal_target, 'Bimodal Target Reference', axes[4, 0], axes[4, 1])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show() 