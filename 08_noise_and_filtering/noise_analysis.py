import matplotlib.pyplot as plt
import numpy as np
import os
import imageio.v2 as imageio
from PIL import Image

def hist_stretch(I):
    """
    Histogram stretching to range [0;1]
    """
    I = I - np.min(I)
    I = I / np.max(I) if np.max(I) > 0 else I
    return I

def ensure_dir(dir_path):
    """
    Create directory if it doesn't exist
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def noise_estimation():
    """
    Perform noise estimation on image regions of interest
    """
    # Ensure output directory exists
    plots_dir = 'plots/TP8_Compression'
    ensure_dir(plots_dir)
    
    # Load the leg image
    img_path = 'images/jambe.tif'
    try:
        # Try reading with imageio
        A = imageio.imread(img_path)
    except:
        # Fallback to PIL if imageio fails
        A = np.array(Image.open(img_path))
    
    # Convert to grayscale if it's RGB
    if len(A.shape) > 2:
        A = np.mean(A, axis=2).astype(np.uint8)
    
    # 1. Extract ROI from a uniform intensity region and plot histogram
    roi = A[160:200, 200:240]
    
    plt.figure(figsize=(12, 10))
    plt.suptitle('Histogram of the Region of Interest (ROI)', fontsize=16)
    
    # Show the original image and mark ROI
    plt.subplot(221)
    plt.imshow(A, cmap='gray')
    plt.title('Original Image')
    # Mark the ROI with a red rectangle
    x, y = 200, 160
    w, h = 40, 40
    plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], 'r-', linewidth=2)
    plt.axis('off')
    
    # Show the ROI
    plt.subplot(222)
    plt.imshow(roi, cmap='gray')
    plt.title('Region of Interest (ROI)')
    plt.axis('off')
    
    # Plot histogram of ROI
    plt.subplot(223)
    plt.hist(roi.flatten(), bins=255, color='blue', alpha=0.7)
    plt.title('Histogram of ROI')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plots_dir, 'histogram_roi.png'), dpi=300)
    plt.show()
    
    # 2. Add exponential noise to the image
    nx, ny = A.shape
    expnoise = -1/0.5 * np.log(1 - np.random.rand(nx, ny))
    expnoise = expnoise / np.max(expnoise)
    B_exp = A + 255 * expnoise
    B_exp = np.clip(B_exp, 0, 255).astype(np.uint8)
    
    # Extract the same ROI from the exponential noise image
    roi_exp = B_exp[160:200, 200:240]
    
    # Plot the exponential noise results
    plt.figure(figsize=(12, 10))
    plt.suptitle('Image with Exponential Noise', fontsize=16)
    
    # Show the exponential noise image and mark ROI
    plt.subplot(221)
    plt.imshow(B_exp, cmap='gray')
    plt.title('Image with Exponential Noise')
    # Mark the ROI with a red rectangle
    plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], 'r-', linewidth=2)
    plt.axis('off')
    
    # Show the ROI of exponential noise image
    plt.subplot(222)
    plt.imshow(roi_exp, cmap='gray')
    plt.title('ROI with Exponential Noise')
    plt.axis('off')
    
    # Plot histogram of ROI with exponential noise
    plt.subplot(223)
    plt.hist(roi_exp.flatten(), bins=255, color='red', alpha=0.7)
    plt.title('Histogram of ROI with Exponential Noise')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plots_dir, 'histogram_exp_noise.png'), dpi=300)
    plt.show()
    
    # Save the exponential noise image
    imageio.imwrite(os.path.join(plots_dir, 'leg_exponential.png'), B_exp)
    
    # 3. Add Gaussian noise to the image
    gaussnoise = 50 * np.random.randn(nx, ny)
    B_gauss = A + gaussnoise
    B_gauss = np.clip(B_gauss, 0, 255).astype(np.uint8)
    
    # Extract the same ROI from the gaussian noise image
    roi_gauss = B_gauss[160:200, 200:240]
    
    # Plot the gaussian noise results
    plt.figure(figsize=(12, 10))
    plt.suptitle('Image with Gaussian Noise', fontsize=16)
    
    # Show the gaussian noise image and mark ROI
    plt.subplot(221)
    plt.imshow(B_gauss, cmap='gray')
    plt.title('Image with Gaussian Noise')
    # Mark the ROI with a red rectangle
    plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], 'r-', linewidth=2)
    plt.axis('off')
    
    # Show the ROI of gaussian noise image
    plt.subplot(222)
    plt.imshow(roi_gauss, cmap='gray')
    plt.title('ROI with Gaussian Noise')
    plt.axis('off')
    
    # Plot histogram of ROI with gaussian noise
    plt.subplot(223)
    plt.hist(roi_gauss.flatten(), bins=255, color='green', alpha=0.7)
    plt.title('Histogram of ROI with Gaussian Noise')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plots_dir, 'histogram_gauss_noise.png'), dpi=300)
    plt.show()
    
    # Save the gaussian noise image
    imageio.imwrite(os.path.join(plots_dir, 'leg_gaussian.png'), B_gauss)
    
    print("Noise estimation completed!")
    print(f"Results saved in {plots_dir}")
    print("Generated files:")
    print("- histogram_roi.png: Original ROI and its histogram")
    print("- histogram_exp_noise.png: ROI with exponential noise and its histogram")
    print("- histogram_gauss_noise.png: ROI with Gaussian noise and its histogram")
    print("- leg_exponential.png: Full image with exponential noise")
    print("- leg_gaussian.png: Full image with Gaussian noise")

if __name__ == '__main__':
    noise_estimation() 