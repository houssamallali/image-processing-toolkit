import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage import exposure
from skimage.io import imread

def create_bimodal_distribution(size=256, mode1=40, mode2=170, spread1=15, spread2=25, weight1=0.3):
    """Create a bimodal distribution to use as a target histogram."""
    x = np.arange(size)
    y1 = np.exp(-(x - mode1)**2 / (2 * spread1**2))
    y2 = np.exp(-(x - mode2)**2 / (2 * spread2**2))
    y = weight1 * y1 + (1 - weight1) * y2
    return y / np.sum(y)

def match_histogram_manual(source, target_hist):
    """
    Match the histogram of an image to a target histogram.
    
    Parameters:
    -----------
    source : ndarray
        The source image to be modified
    target_hist : ndarray
        The target histogram (normalized)
        
    Returns:
    --------
    matched : ndarray
        The modified image with histogram matched to target
    """
    # Get the source histogram and calculate CDFs
    src_values, src_counts = np.unique(source.ravel(), return_counts=True)
    src_quantiles = np.cumsum(src_counts).astype(np.float64)
    src_quantiles /= src_quantiles[-1]
    
    # Calculate the target CDF
    target_quantiles = np.cumsum(target_hist)
    
    # Create the mapping
    interp_values = np.interp(src_quantiles, target_quantiles, np.arange(len(target_quantiles)))
    
    # Map the source values to the new values
    mapping_func = lambda x: interp_values[np.searchsorted(src_values, x)]
    
    # Apply the mapping to each pixel in the source image
    matched = np.vectorize(mapping_func)(source)
    
    # Normalize to [0, 1]
    matched = (matched - matched.min()) / (matched.max() - matched.min())
    
    return matched

def main():
    # Load the phobos image directly since it already exists (as JPG, not PNG)
    img = imread('../images/phobos.jpg', as_gray=True) / 255.0

    # Create a bimodal target distribution
    target_hist = create_bimodal_distribution(size=256, mode1=60, mode2=180, weight1=0.4)
    
    # Perform histogram equalization
    img_eq = exposure.equalize_hist(img)
    
    # Perform manual histogram matching
    img_matched_manual = match_histogram_manual(img, target_hist)
    
    # Create a reference image for skimage's match_histograms
    # Create a reference array with the same number of dimensions as the source
    reference = np.linspace(0, 1, 256).reshape(-1, 1)
    # Use the target histogram to generate pixel values
    reference_values = np.random.choice(
        np.linspace(0, 1, 256), 
        size=10000, 
        p=target_hist
    ).reshape(100, 100)
    
    # Perform histogram matching using skimage
    img_matched_ski = exposure.match_histograms(img, reference_values)

    # Plot the results
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    
    # Original image
    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(img, cmap='gray')
    ax0.set_title('(a) Original Image')
    ax0.axis('off')
    
    # Histogram equalization result
    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(img_eq, cmap='gray')
    ax1.set_title('(b) Histogram Equalization')
    ax1.axis('off')
    
    # Histogram matching result (manual)
    ax2 = plt.subplot(gs[0, 2])
    ax2.imshow(img_matched_manual, cmap='gray')
    ax2.set_title('(c) Histogram Matching (Manual)')
    ax2.axis('off')
    
    # Histogram matching result (skimage)
    ax3 = plt.subplot(gs[0, 3])
    ax3.imshow(img_matched_ski, cmap='gray')
    ax3.set_title('(d) Histogram Matching (skimage)')
    ax3.axis('off')
    
    # Original histogram
    ax4 = plt.subplot(gs[1, 0])
    ax4.hist(img.ravel(), bins=256, range=(0, 1), density=True, color='black', alpha=0.7)
    ax4.set_title('(e) Original Histogram')
    ax4.set_xlim(0, 1)
    
    # Equalized histogram
    ax5 = plt.subplot(gs[1, 1])
    ax5.hist(img_eq.ravel(), bins=256, range=(0, 1), density=True, color='black', alpha=0.7)
    ax5.set_title('(f) Equalized Histogram')
    ax5.set_xlim(0, 1)
    
    # Target histogram
    ax6 = plt.subplot(gs[1, 2])
    ax6.plot(np.linspace(0, 1, 256), target_hist * 256, 'k-')
    ax6.set_title('(g) Target Distribution')
    ax6.set_xlim(0, 1)
    
    # Matched histograms
    ax7 = plt.subplot(gs[1, 3])
    ax7.hist(img_matched_manual.ravel(), bins=256, range=(0, 1), density=True, 
             color='blue', alpha=0.5, label='Manual')
    ax7.hist(img_matched_ski.ravel(), bins=256, range=(0, 1), density=True, 
             color='red', alpha=0.5, label='skimage')
    ax7.set_title('(h) Matched Histograms')
    ax7.set_xlim(0, 1)
    ax7.legend()
    
    plt.tight_layout()
    plt.savefig('../TP5_Enhancement/histogram_matching_phobos.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main() 