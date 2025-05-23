import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from astropy.io import fits
import os

def checkerboard(s=8):
    """
    Generate a checkerboard pattern
    
    Parameters:
    -----------
    s : int
        Size of the checkerboard (number of squares per side)
        
    Returns:
    --------
    ndarray
        Checkerboard image
    """
    return np.kron(([1, 0] * 4, [0, 1] * 4) * 4, np.ones((s, s)))

def gaussian_psf(size=15, sigma=2):
    """
    Generate a Gaussian PSF
    
    Parameters:
    -----------
    size : int
        Size of the PSF kernel
    sigma : float
        Standard deviation of the Gaussian
        
    Returns:
    --------
    ndarray
        Gaussian PSF
    """
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = size // 2, size // 2
    
    # Create Gaussian kernel
    psf = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
    
    # Normalize to sum to 1
    return psf / psf.sum()

def motion_psf(length=15, angle=45):
    """
    Generate a motion blur PSF
    
    Parameters:
    -----------
    length : int
        Length of the motion blur
    angle : float
        Angle of the motion blur in degrees
        
    Returns:
    --------
    ndarray
        Motion blur PSF
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Create PSF kernel
    size = int(length * 1.5)
    psf = np.zeros((size, size))
    center = size // 2
    
    # Create the motion blur effect
    for i in range(length):
        offset = i - length // 2
        x = center + int(np.round(offset * np.cos(angle_rad)))
        y = center + int(np.round(offset * np.sin(angle_rad)))
        if 0 <= x < size and 0 <= y < size:
            psf[y, x] = 1
    
    # Normalize to sum to 1
    if psf.sum() > 0:
        psf /= psf.sum()
    
    return psf

def add_gaussian_noise(image, sigma=0.01):
    """
    Add Gaussian noise to an image
    
    Parameters:
    -----------
    image : ndarray
        Input image
    sigma : float
        Noise level (standard deviation)
        
    Returns:
    --------
    ndarray
        Noisy image
    """
    # Create a copy to avoid modifying the original
    noisy_image = image.copy()
    
    # Get image range
    img_min = np.min(image)
    img_max = np.max(image)
    img_range = img_max - img_min
    
    # Add noise
    noise = np.random.normal(0, sigma * img_range, image.shape)
    noisy_image = noisy_image + noise
    
    # Clip values to maintain original range
    noisy_image[noisy_image > img_max] = img_max
    noisy_image[noisy_image < img_min] = img_min
    
    return noisy_image

def psf2otf(psf, shape):
    """
    Convert PSF to OTF (Optical Transfer Function)
    
    Parameters:
    -----------
    psf : ndarray
        Point Spread Function
    shape : tuple
        Shape of the output OTF
        
    Returns:
    --------
    ndarray
        Optical Transfer Function (frequency domain representation of PSF)
    """
    # Convert inputs to arrays
    psf = np.array(psf)
    shape = np.array(shape)
    
    # Padding to match target shape
    pad_size = shape - np.array(psf.shape)
    pad = [(0, p) for p in pad_size]
    padded_psf = np.pad(psf, pad, mode='constant')
    
    # Shift the PSF to center it
    shift = (np.array(padded_psf.shape) // 2).astype(int)
    psf_centered = np.roll(padded_psf, shift, axis=(0, 1))
    
    # Calculate OTF (FFT of centered PSF)
    otf = np.fft.fft2(psf_centered)
    
    # Return real part (OTF should be real for symmetric PSF)
    return np.real(otf)

def inverse_filter(blurred_img, psf, alpha=0.001):
    """
    Apply inverse filtering in frequency domain
    
    Parameters:
    -----------
    blurred_img : ndarray
        Blurred image
    psf : ndarray
        Point Spread Function
    alpha : float
        Regularization parameter to avoid division by zero
        
    Returns:
    --------
    ndarray
        Restored image
    """
    # Convert PSF to OTF
    otf = psf2otf(psf, blurred_img.shape)
    
    # Compute FFT of blurred image
    blurred_fft = np.fft.fft2(blurred_img)
    
    # Perform inverse filtering with regularization
    restored_fft = blurred_fft / (otf + alpha)
    
    # Convert back to spatial domain
    restored = np.real(np.fft.ifft2(restored_fft))
    
    return restored

def wiener_filter(blurred_noisy_img, psf, K=0.01, estimate_noise=False, original_img=None):
    """
    Apply Wiener filtering for restoration
    
    Parameters:
    -----------
    blurred_noisy_img : ndarray
        Blurred and noisy image
    psf : ndarray
        Point Spread Function
    K : float
        Noise-to-signal ratio parameter
    estimate_noise : bool
        Whether to estimate K from the images
    original_img : ndarray, optional
        Original image for comparison (only used if estimate_noise=True)
        
    Returns:
    --------
    ndarray
        Restored image
    """
    # Convert PSF to OTF
    otf = psf2otf(psf, blurred_noisy_img.shape)
    
    # Compute FFT of blurred image
    blurred_fft = np.fft.fft2(blurred_noisy_img)
    
    if estimate_noise and original_img is not None:
        # Estimate noise and image power
        original_fft = np.fft.fft2(original_img)
        noise_power = np.mean(np.abs(blurred_fft - original_fft * otf)**2)
        signal_power = np.mean(np.abs(original_fft)**2)
        
        # Use estimated noise-to-signal ratio
        K = noise_power / signal_power
    
    # Apply Wiener filter
    conj_otf = np.conjugate(otf)
    wiener_fft = conj_otf / (np.abs(otf)**2 + K) * blurred_fft
    
    # Convert back to spatial domain
    restored = np.real(np.fft.ifft2(wiener_fft))
    
    return restored

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists('../plots/TP6_Restoration'):
        os.makedirs('../plots/TP6_Restoration')
    
    # 1. Generate the checkerboard image
    print("Generating checkerboard image...")
    cb = checkerboard(8)
    
    # 2. Create a PSF (Point Spread Function)
    print("Creating PSF...")
    # Option 1: Gaussian PSF
    psf_gaussian = gaussian_psf(size=15, sigma=2)
    
    # Option 2: Motion PSF
    psf_motion = motion_psf(length=15, angle=45)
    
    # Choose which PSF to use
    psf = psf_motion  # Change to psf_gaussian to use Gaussian PSF
    
    # 3. Apply PSF to checkerboard (create a damaged/blurred image)
    print("Applying blur to image...")
    blurred_cb = signal.convolve2d(cb, psf, boundary='wrap', mode='same')
    
    # 4. Add Gaussian noise to the blurred image
    print("Adding Gaussian noise...")
    sigma_noise = 0.02  # Adjust noise level
    blurred_noisy_cb = add_gaussian_noise(blurred_cb, sigma=sigma_noise)
    
    # 5. Apply inverse filtering in the Fourier domain
    print("Applying inverse filtering...")
    # For the no-noise case
    restored_inv = inverse_filter(blurred_cb, psf, alpha=0.001)
    
    # For the noisy case
    restored_inv_noisy = inverse_filter(blurred_noisy_cb, psf, alpha=0.001)
    
    # 6. Apply Wiener filter for better noise handling
    print("Applying Wiener filtering...")
    restored_wiener = wiener_filter(blurred_noisy_cb, psf, K=0.01)
    
    # 7. Visualize results
    print("Visualizing results...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original checkerboard
    axes[0, 0].imshow(cb, cmap='gray')
    axes[0, 0].set_title('Original Checkerboard')
    axes[0, 0].axis('off')
    
    # PSF
    axes[0, 1].imshow(psf, cmap='viridis')
    axes[0, 1].set_title('Point Spread Function (PSF)')
    axes[0, 1].axis('off')
    
    # Blurred image
    axes[0, 2].imshow(blurred_cb, cmap='gray')
    axes[0, 2].set_title('Blurred Image (no noise)')
    axes[0, 2].axis('off')
    
    # Blurred and noisy image
    axes[1, 0].imshow(blurred_noisy_cb, cmap='gray')
    axes[1, 0].set_title('Blurred + Noisy Image')
    axes[1, 0].axis('off')
    
    # Inverse filter (noisy)
    axes[1, 1].imshow(np.clip(restored_inv_noisy, 0, 1), cmap='gray')
    axes[1, 1].set_title('Inverse Filter (with noise)')
    axes[1, 1].axis('off')
    
    # Wiener filter
    axes[1, 2].imshow(np.clip(restored_wiener, 0, 1), cmap='gray')
    axes[1, 2].set_title('Wiener Filter Restoration')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('../plots/TP6_Restoration/restoration_comparison.png', dpi=300)
    plt.show()
    
    # FFT Visualization
    plt.figure(figsize=(15, 5))
    
    # OTF (frequency domain representation of PSF)
    otf = psf2otf(psf, cb.shape)
    plt.subplot(131)
    plt.imshow(np.log(1 + np.abs(np.fft.fftshift(otf))), cmap='viridis')
    plt.title('OTF (Optical Transfer Function)')
    plt.colorbar()
    
    # FFT of original image
    cb_fft = np.fft.fft2(cb)
    plt.subplot(132)
    plt.imshow(np.log(1 + np.abs(np.fft.fftshift(cb_fft))), cmap='viridis')
    plt.title('FFT of Original Image')
    plt.colorbar()
    
    # FFT of blurred image
    blurred_fft = np.fft.fft2(blurred_cb)
    plt.subplot(133)
    plt.imshow(np.log(1 + np.abs(np.fft.fftshift(blurred_fft))), cmap='viridis')
    plt.title('FFT of Blurred Image')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('../plots/TP6_Restoration/fft_visualization.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main() 