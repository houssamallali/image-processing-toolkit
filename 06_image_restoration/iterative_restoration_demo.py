import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import os
import time
from skimage import data

def richardson_lucy_deconvolution(blurred_img, psf, num_iterations=10, verbose=False):
    """
    Richardson-Lucy deconvolution algorithm for image restoration
    
    Parameters:
    -----------
    blurred_img : ndarray
        Blurred image to restore
    psf : ndarray
        Point Spread Function
    num_iterations : int
        Number of iterations to perform
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    ndarray
        Restored image
    """
    # Make sure PSF is normalized
    psf = psf / np.sum(psf)
    
    # Initialize the estimate with the blurred image
    estimate = np.copy(blurred_img)
    
    # Flip PSF for convolution (equivalent to correlation)
    psf_flipped = np.flip(psf)
    
    # Iterate
    start_time = time.time()
    for i in range(num_iterations):
        if verbose and (i % 2 == 0 or i == num_iterations - 1):
            print(f"Richardson-Lucy iteration {i+1}/{num_iterations}")
            
        # Calculate current estimate convolved with PSF
        reblurred = signal.convolve2d(estimate, psf, mode='same', boundary='wrap')
        
        # Avoid division by zero
        reblurred[reblurred == 0] = np.finfo(float).eps
        
        # Calculate ratio of original blurred image to re-blurred estimate
        ratio = blurred_img / reblurred
        
        # Convolve the ratio with the flipped PSF
        correction = signal.convolve2d(ratio, psf_flipped, mode='same', boundary='wrap')
        
        # Update estimate
        estimate *= correction
    
    # Ensure values are within valid range
    estimate = np.clip(estimate, 0, 1)
    
    if verbose:
        print(f"Richardson-Lucy completed in {time.time() - start_time:.2f} seconds")
    
    return estimate

def van_cittert_deconvolution(blurred_img, psf, num_iterations=10, beta=0.01, verbose=False):
    """
    Van Cittert iterative deconvolution algorithm for image restoration
    
    Parameters:
    -----------
    blurred_img : ndarray
        Blurred image to restore
    psf : ndarray
        Point Spread Function
    num_iterations : int
        Number of iterations to perform
    beta : float
        Relaxation parameter controlling convergence speed
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    ndarray
        Restored image
    """
    # Make sure PSF is normalized
    psf = psf / np.sum(psf)
    
    # Initialize the estimate with the blurred image
    estimate = np.copy(blurred_img)
    
    # Iterate
    start_time = time.time()
    for i in range(num_iterations):
        if verbose and (i % 2 == 0 or i == num_iterations - 1):
            print(f"Van-Cittert iteration {i+1}/{num_iterations}")
            
        # Calculate current estimate convolved with PSF
        reblurred = signal.convolve2d(estimate, psf, mode='same', boundary='wrap')
        
        # Calculate residual (difference between original and re-blurred image)
        residual = blurred_img - reblurred
        
        # Update estimate: f_{k+1} = f_k + β(g - h * f_k)
        estimate = estimate + beta * residual
        
        # Ensure values are within valid range after each iteration
        estimate = np.clip(estimate, 0, 1)
    
    if verbose:
        print(f"Van-Cittert completed in {time.time() - start_time:.2f} seconds")
    
    return estimate

def main():
    """
    Test function for deconvolution algorithms on a small test image
    """
    # Create output directory if it doesn't exist
    output_dir = '../plots/TP6_Restoration'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Testing deconvolution algorithms...")
    
    # Create small test image (64x64)
    camera = data.camera()[::8, ::8] / 255.0
    print(f"Test image shape: {camera.shape}")
    
    # Create simple Gaussian PSF
    psf = np.zeros((5, 5))
    psf[2, 2] = 1
    psf = ndimage.gaussian_filter(psf, sigma=1)
    psf /= psf.sum()
    
    # Create blurred version
    print("Creating blurred image...")
    blurred = signal.convolve2d(camera, psf, mode='same', boundary='wrap')
    
    # Apply Richardson-Lucy restoration
    print("Applying Richardson-Lucy deconvolution...")
    start_time = time.time()
    restored_rl = richardson_lucy_deconvolution(blurred, psf, num_iterations=10, verbose=True)
    print(f"Total Richardson-Lucy time: {time.time() - start_time:.2f} seconds")
    
    # Apply Van-Cittert restoration
    print("Applying Van-Cittert deconvolution...")
    start_time = time.time()
    restored_vc = van_cittert_deconvolution(blurred, psf, num_iterations=10, beta=0.01, verbose=True)
    print(f"Total Van-Cittert time: {time.time() - start_time:.2f} seconds")
    
    # Save Richardson-Lucy results
    print("Saving Richardson-Lucy results...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(camera, cmap='gray')
    plt.title('Original Test Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(restored_rl, cmap='gray')
    plt.title('Richardson-Lucy (10 iter)')
    plt.axis('off')
    
    plt.tight_layout()
    output_path = f'{output_dir}/test_richardson_lucy.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Richardson-Lucy result to {output_path}")
    
    # Save Van-Cittert results
    print("Saving Van-Cittert results...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(camera, cmap='gray')
    plt.title('Original Test Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(restored_vc, cmap='gray')
    plt.title('Van-Cittert (10 iter)')
    plt.axis('off')
    
    plt.tight_layout()
    output_path = f'{output_dir}/test_van_cittert.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Van-Cittert result to {output_path}")
    
    # Save comparison
    print("Saving comparison results...")
    plt.figure(figsize=(15, 3))
    
    plt.subplot(141)
    plt.imshow(camera, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(restored_rl, cmap='gray')
    plt.title('Richardson-Lucy')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(restored_vc, cmap='gray')
    plt.title('Van-Cittert')
    plt.axis('off')
    
    plt.tight_layout()
    output_path = f'{output_dir}/test_comparison.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved comparison result to {output_path}")
    
    print("Test completed successfully.")

if __name__ == "__main__":
    main() 