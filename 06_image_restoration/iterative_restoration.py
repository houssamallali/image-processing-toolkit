import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import os
from matplotlib.colors import PowerNorm
import time
from skimage.transform import resize

def convert_to_grayscale(image):
    """Convert RGB image to grayscale if needed"""
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return np.mean(image[:,:,:3], axis=2)
    return image

def downsample_image(image, max_size=256):
    """
    Downsample an image if it's too large
    
    Parameters:
    -----------
    image : ndarray
        Input image
    max_size : int
        Maximum dimension size
        
    Returns:
    --------
    ndarray
        Downsampled image
    """
    # Check if downsizing is needed
    if max(image.shape[:2]) <= max_size:
        return image
        
    # Calculate scale factor
    scale = max_size / max(image.shape[:2])
    
    # Handle RGB vs grayscale
    if len(image.shape) == 3:
        new_shape = (int(image.shape[0] * scale), int(image.shape[1] * scale), image.shape[2])
    else:
        new_shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
    
    # Resize image
    return resize(image, new_shape, anti_aliasing=True, preserve_range=True)

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
    # Ensure inputs are 2D arrays
    blurred_img_2d = convert_to_grayscale(blurred_img)
    psf_2d = convert_to_grayscale(psf)
    
    # Make sure PSF is normalized
    psf_2d = psf_2d / np.sum(psf_2d)
    
    # Initialize the estimate with the blurred image
    estimate = np.copy(blurred_img_2d)
    
    # Flip PSF for convolution (equivalent to correlation)
    psf_flipped = np.flip(psf_2d)
    
    # Iterate
    start_time = time.time()
    for i in range(num_iterations):
        if verbose and (i % 5 == 0 or i == num_iterations - 1):
            print(f"Richardson-Lucy iteration {i+1}/{num_iterations}")
            
        # Calculate current estimate convolved with PSF
        reblurred = signal.convolve2d(estimate, psf_2d, mode='same', boundary='wrap')
        
        # Avoid division by zero
        reblurred[reblurred == 0] = np.finfo(float).eps
        
        # Calculate ratio of original blurred image to re-blurred estimate
        ratio = blurred_img_2d / reblurred
        
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
    # Ensure inputs are 2D arrays
    blurred_img_2d = convert_to_grayscale(blurred_img)
    psf_2d = convert_to_grayscale(psf)
    
    # Make sure PSF is normalized
    psf_2d = psf_2d / np.sum(psf_2d)
    
    # Initialize the estimate with the blurred image
    estimate = np.copy(blurred_img_2d)
    
    # Iterate
    start_time = time.time()
    for i in range(num_iterations):
        if verbose and (i % 5 == 0 or i == num_iterations - 1):
            print(f"Van-Cittert iteration {i+1}/{num_iterations}")
            
        # Calculate current estimate convolved with PSF
        reblurred = signal.convolve2d(estimate, psf_2d, mode='same', boundary='wrap')
        
        # Calculate residual (difference between original and re-blurred image)
        residual = blurred_img_2d - reblurred
        
        # Update estimate: f_{k+1} = f_k + β(g - h * f_k)
        estimate = estimate + beta * residual
        
        # Ensure values are within valid range after each iteration
        estimate = np.clip(estimate, 0, 1)
    
    if verbose:
        print(f"Van-Cittert completed in {time.time() - start_time:.2f} seconds")
    
    return estimate

def landweber_deconvolution(blurred_img, psf, num_iterations=50, alpha=1.0, verbose=False):
    """
    Landweber iterative deconvolution algorithm for image restoration
    
    Parameters:
    -----------
    blurred_img : ndarray
        Blurred image to restore
    psf : ndarray
        Point Spread Function
    num_iterations : int
        Number of iterations to perform (reduced from 200 to 50 for performance)
    alpha : float
        Relaxation parameter (Jansson parameter)
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    ndarray
        Restored image
    """
    # Ensure inputs are 2D arrays
    blurred_img_2d = convert_to_grayscale(blurred_img)
    psf_2d = convert_to_grayscale(psf)
    
    # Make sure PSF is normalized
    psf_2d = psf_2d / np.sum(psf_2d)
    
    # Initialize the estimate with the blurred image
    estimate = np.copy(blurred_img_2d)
    
    # Flip PSF for convolution
    psf_flipped = np.flip(psf_2d)
    
    # Calculate largest eigenvalue of H^T H to determine optimal step size
    psf_fft = np.fft.fft2(psf_2d, s=blurred_img_2d.shape)
    psf_conj_fft = np.conjugate(psf_fft)
    largest_eig = np.max(np.abs(psf_fft * psf_conj_fft))
    
    # Calculate relaxation parameter
    relaxation = alpha / largest_eig if largest_eig > 0 else alpha
    
    # Iterate
    start_time = time.time()
    for i in range(num_iterations):
        if verbose and (i % 10 == 0 or i == num_iterations - 1):
            print(f"Landweber iteration {i+1}/{num_iterations}")
            
        # Calculate current estimate convolved with PSF
        reblurred = signal.convolve2d(estimate, psf_2d, mode='same', boundary='wrap')
        
        # Calculate residual (difference between original and re-blurred image)
        residual = blurred_img_2d - reblurred
        
        # Calculate correction term (H^T * residual)
        correction = signal.convolve2d(residual, psf_flipped, mode='same', boundary='wrap')
        
        # Update estimate with Landweber iteration formula
        estimate = estimate + relaxation * correction
        
        # Apply positivity constraint
        estimate = np.clip(estimate, 0, 1)
    
    if verbose:
        print(f"Landweber completed in {time.time() - start_time:.2f} seconds")
    
    return estimate

def apply_iterative_restoration(image, psf, methods=None, verbose=False):
    """
    Apply various iterative restoration methods to an image
    
    Parameters:
    -----------
    image : ndarray
        Input blurred image
    psf : ndarray
        Point Spread Function
    methods : list
        List of methods to apply, defaults to all available methods
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict
        Dictionary of restored images for each method
    """
    if methods is None:
        methods = ['richardson_lucy', 'van_cittert', 'landweber']
    
    results = {}
    
    # Apply each method
    for method in methods:
        if verbose:
            print(f"\nApplying {method} method...")
            
        if method == 'richardson_lucy':
            results[method] = richardson_lucy_deconvolution(image, psf, num_iterations=10, verbose=verbose)
        elif method == 'van_cittert':
            results[method] = van_cittert_deconvolution(image, psf, num_iterations=10, beta=0.01, verbose=verbose)
        elif method == 'landweber':
            results[method] = landweber_deconvolution(image, psf, num_iterations=50, alpha=1.0, verbose=verbose)
    
    return results

def compare_iterative_methods(image, psf, title='', verbose=False, max_size=256):
    """
    Compare different iterative restoration methods and display results
    
    Parameters:
    -----------
    image : ndarray
        Input blurred image
    psf : ndarray
        Point Spread Function
    title : str
        Title prefix for the comparison plot
    verbose : bool
        Whether to print progress information
    max_size : int
        Maximum dimension size for processing
    """
    print(f"\nProcessing {title} image...")
    print(f"Original image shape: {image.shape}")
    
    # Downsample if needed
    image_resized = downsample_image(image, max_size=max_size)
    psf_resized = downsample_image(psf, max_size=min(64, max_size // 4))
    
    if image.shape != image_resized.shape:
        print(f"Downsampled to shape: {image_resized.shape}")
    
    # Convert image to grayscale for display consistency
    image_gray = convert_to_grayscale(image_resized)
    
    # Apply all iterative methods
    results = apply_iterative_restoration(image_resized, psf_resized, verbose=verbose)
    
    # Display comparison
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(231)
    plt.imshow(image_gray, cmap='gray')
    plt.title(f'Original {title}')
    plt.axis('off')
    
    # Direct deconvolution - use a high alpha to avoid extreme noise
    from astronomy_restoration import inverse_filter
    direct = inverse_filter(image_gray, convert_to_grayscale(psf_resized), alpha=0.01)
    plt.subplot(232)
    plt.imshow(direct, cmap='gray')
    plt.title('Direct Deconvolution')
    plt.axis('off')
    
    # Richardson-Lucy
    plt.subplot(233)
    plt.imshow(results['richardson_lucy'], cmap='gray')
    plt.title('Richardson-Lucy (10 iter)')
    plt.axis('off')
    
    # Van-Cittert
    plt.subplot(234)
    plt.imshow(results['van_cittert'], cmap='gray')
    plt.title('Van-Cittert (10 iter)')
    plt.axis('off')
    
    # Landweber
    plt.subplot(235)
    plt.imshow(results['landweber'], cmap='gray')
    plt.title('Landweber (50 iter, α=1)')
    plt.axis('off')
    
    # Wiener filter for comparison
    from astronomy_restoration import wiener_filter
    wiener = wiener_filter(image_gray, convert_to_grayscale(psf_resized), K=0.01)
    plt.subplot(236)
    plt.imshow(wiener, cmap='gray')
    plt.title('Wiener Filter')
    plt.axis('off')
    
    plt.tight_layout()
    output_path = f'../plots/TP6_Restoration/{title}_iterative_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved comparison image to {output_path}")
    plt.show()

def main():
    """
    Main function to test iterative restoration methods on astronomical images
    """
    # Create output directory if it doesn't exist
    if not os.path.exists('../plots/TP6_Restoration'):
        os.makedirs('../plots/TP6_Restoration')
    
    # Set verbosity and maximum image size
    verbose = True
    max_size = 128  # Maximum dimension in pixels (reduced for faster testing)
    
    # Only run Richardson-Lucy for now
    test_methods = ['richardson_lucy']
    
    # Import astronomical images
    try:
        from astronomy_restoration import load_astronomical_images
        images = load_astronomical_images()
        
        if images is not None:
            # Process only Jupiter to start with
            # Convert image to grayscale for display consistency
            jupiter = images['jupiter']
            jupiter_psf = images['jupiter_psf']
            
            print(f"\nProcessing Jupiter image...")
            print(f"Original image shape: {jupiter.shape}")
            
            # Downsample if needed
            jupiter_resized = downsample_image(jupiter, max_size=max_size)
            psf_resized = downsample_image(jupiter_psf, max_size=min(32, max_size // 4))
            
            if jupiter.shape != jupiter_resized.shape:
                print(f"Downsampled to shape: {jupiter_resized.shape}")
            
            # Convert to grayscale
            jupiter_gray = convert_to_grayscale(jupiter_resized)
            psf_gray = convert_to_grayscale(psf_resized)
            
            # Apply Richardson-Lucy algorithm
            print("\nApplying Richardson-Lucy method...")
            start_time = time.time()
            restored = richardson_lucy_deconvolution(jupiter_gray, psf_gray, num_iterations=10, verbose=verbose)
            print(f"Total time: {time.time() - start_time:.2f} seconds")
            
            # Display results
            plt.figure(figsize=(12, 4))
            
            plt.subplot(131)
            plt.imshow(jupiter_gray, cmap='gray')
            plt.title('Original Jupiter')
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(psf_gray, cmap='gray')
            plt.title('Jupiter PSF')
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(restored, cmap='gray')
            plt.title('Richardson-Lucy (10 iter)')
            plt.axis('off')
            
            plt.tight_layout()
            output_path = f'../plots/TP6_Restoration/jupiter_richardson_lucy.png'
            plt.savefig(output_path, dpi=300)
            print(f"Saved result to {output_path}")
            plt.show()
        else:
            print("Failed to load astronomical images.")
    except ImportError as e:
        print(f"Error importing astronomy_restoration: {e}")
        print("Creating test images instead...")
        
        # Create test images if astronomical images not available
        from skimage import data
        
        # Create test image using camera image (smaller size)
        camera = data.camera()[::4, ::4] / 255.0  # Downsample for speed
        
        # Create simple Gaussian PSF
        psf = np.zeros((7, 7))
        psf[3, 3] = 1
        psf = ndimage.gaussian_filter(psf, sigma=1)
        psf /= psf.sum()
        
        # Create blurred version
        blurred = signal.convolve2d(camera, psf, mode='same', boundary='wrap')
        
        # Apply Richardson-Lucy restoration
        print("\nApplying Richardson-Lucy method to test image...")
        start_time = time.time()
        restored = richardson_lucy_deconvolution(blurred, psf, num_iterations=10, verbose=verbose)
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        
        # Display results
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
        plt.imshow(restored, cmap='gray')
        plt.title('Richardson-Lucy (10 iter)')
        plt.axis('off')
        
        plt.tight_layout()
        output_path = f'../plots/TP6_Restoration/test_richardson_lucy.png'
        plt.savefig(output_path, dpi=300)
        print(f"Saved result to {output_path}")
        plt.show()

if __name__ == "__main__":
    main() 