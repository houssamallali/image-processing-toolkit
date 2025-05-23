import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from astropy.io import fits
import os
from skimage import img_as_float, io, color
from matplotlib.colors import PowerNorm

# Define our own restoration functions instead of importing from image_restoration
def convert_to_grayscale(image):
    """Convert RGB image to grayscale if needed"""
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return np.mean(image[:,:,:3], axis=2)
    return image

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
        Optical Transfer Function
    """
    # Ensure PSF is 2D
    psf = convert_to_grayscale(psf)
    
    # Convert inputs to arrays
    psf = np.array(psf)
    
    # Get the target shape (first two dimensions)
    target_shape = shape[:2] if len(shape) > 2 else shape
    
    # Padding to match target shape
    padded_psf = np.zeros(target_shape)
    h, w = psf.shape
    h_pad, w_pad = target_shape
    
    # Center PSF in the padded array
    h_off = (h_pad - h) // 2
    w_off = (w_pad - w) // 2
    padded_psf[h_off:h_off+h, w_off:w_off+w] = psf
    
    # Circularly shift PSF to place the origin at (0,0)
    psf_centered = np.roll(padded_psf, (-h_off, -w_off), axis=(0, 1))
    
    # Calculate OTF using FFT
    otf = np.fft.fft2(psf_centered)
    
    return otf

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
        Regularization parameter
        
    Returns:
    --------
    ndarray
        Restored image
    """
    # Handle RGB images
    if len(blurred_img.shape) == 3 and blurred_img.shape[2] >= 3:
        restored = np.zeros_like(blurred_img)
        for i in range(min(3, blurred_img.shape[2])):
            channel = blurred_img[:,:,i]
            restored[:,:,i] = inverse_filter_channel(channel, psf, alpha)
        return restored
    else:
        return inverse_filter_channel(blurred_img, psf, alpha)

def inverse_filter_channel(blurred_channel, psf, alpha=0.001):
    """Apply inverse filter to a single channel"""
    # Convert PSF to OTF
    otf = psf2otf(psf, blurred_channel.shape)
    
    # Compute FFT of blurred image
    blurred_fft = np.fft.fft2(blurred_channel)
    
    # Perform inverse filtering with regularization
    restored_fft = blurred_fft / (otf + alpha)
    
    # Convert back to spatial domain
    restored = np.real(np.fft.ifft2(restored_fft))
    
    # Ensure values are within valid range
    restored = np.clip(restored, 0, 1)
    
    return restored

def wiener_filter(blurred_img, psf, K=0.01):
    """
    Apply Wiener filtering for restoration
    
    Parameters:
    -----------
    blurred_img : ndarray
        Blurred image
    psf : ndarray
        Point Spread Function
    K : float
        Noise-to-signal ratio parameter
        
    Returns:
    --------
    ndarray
        Restored image
    """
    # Handle RGB images
    if len(blurred_img.shape) == 3 and blurred_img.shape[2] >= 3:
        restored = np.zeros_like(blurred_img)
        for i in range(min(3, blurred_img.shape[2])):
            channel = blurred_img[:,:,i]
            restored[:,:,i] = wiener_filter_channel(channel, psf, K)
        return restored
    else:
        return wiener_filter_channel(blurred_img, psf, K)

def wiener_filter_channel(blurred_channel, psf, K=0.01):
    """Apply Wiener filter to a single channel"""
    # Convert PSF to OTF
    otf = psf2otf(psf, blurred_channel.shape)
    
    # Compute FFT of blurred image
    blurred_fft = np.fft.fft2(blurred_channel)
    
    # Apply Wiener filter
    conj_otf = np.conjugate(otf)
    wiener_fft = conj_otf / (np.abs(otf)**2 + K) * blurred_fft
    
    # Convert back to spatial domain
    restored = np.real(np.fft.ifft2(wiener_fft))
    
    # Ensure values are within valid range
    restored = np.clip(restored, 0, 1)
    
    return restored

def restore_astronomical_image(image, psf, method='wiener', k=0.01, alpha=0.001):
    """
    Apply restoration to an astronomical image
    
    Parameters:
    -----------
    image : ndarray
        Input image to restore
    psf : ndarray
        Point Spread Function
    method : str
        Restoration method ('inverse' or 'wiener')
    k : float
        Noise-to-signal ratio for Wiener filter
    alpha : float
        Regularization parameter for inverse filter
        
    Returns:
    --------
    ndarray
        Restored image
    """
    if method == 'inverse':
        restored = inverse_filter(image, psf, alpha=alpha)
    elif method == 'wiener':
        restored = wiener_filter(image, psf, K=k)
    else:
        raise ValueError(f"Unknown restoration method: {method}")
    
    return restored

def load_astronomical_images():
    """
    Load Jupiter and Saturn images with their PSFs
    
    Returns:
    --------
    dict
        Dictionary containing loaded images and PSFs
    """
    # Base directory for astronomy images
    base_dir = '../images/astronomy'
    
    # Check if directory exists, if not create it
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory {base_dir}")
        print("Please place the astronomy images in this directory before running this script.")
        print("Expected files:")
        print("  - jupiter.fits or jupiter.png")
        print("  - jupiter_psf.fits or jupiter_psf.png")
        print("  - saturn.fits or saturn.png")
        print("  - saturn_psf.fits or saturn_psf.png")
        return None
    
    # Initialize dictionary to store images
    images = {}
    
    # Try to load Jupiter image
    jupiter_path = os.path.join(base_dir, 'jupiter.fits')
    jupiter_png_path = os.path.join(base_dir, 'jupiter.png')
    
    if os.path.exists(jupiter_path):
        # Load FITS file
        with fits.open(jupiter_path) as hdul:
            images['jupiter'] = hdul[0].data
            print(f"Loaded Jupiter FITS image from {jupiter_path}")
    elif os.path.exists(jupiter_png_path):
        # Load PNG file
        images['jupiter'] = img_as_float(io.imread(jupiter_png_path))
        print(f"Loaded Jupiter PNG image from {jupiter_png_path}")
    else:
        print(f"Jupiter image not found. Please place it in {base_dir}")
        return None
    
    # Try to load Jupiter PSF
    jupiter_psf_path = os.path.join(base_dir, 'jupiter_psf.fits')
    jupiter_psf_png_path = os.path.join(base_dir, 'jupiter_psf.png')
    
    if os.path.exists(jupiter_psf_path):
        # Load FITS file
        with fits.open(jupiter_psf_path) as hdul:
            images['jupiter_psf'] = hdul[0].data
            print(f"Loaded Jupiter PSF from {jupiter_psf_path}")
    elif os.path.exists(jupiter_psf_png_path):
        # Load PNG file
        images['jupiter_psf'] = img_as_float(io.imread(jupiter_psf_png_path))
        print(f"Loaded Jupiter PSF from {jupiter_psf_png_path}")
    else:
        print(f"Jupiter PSF not found. Please place it in {base_dir}")
        return None
    
    # Try to load Saturn image
    saturn_path = os.path.join(base_dir, 'saturn.fits')
    saturn_png_path = os.path.join(base_dir, 'saturn.png')
    
    if os.path.exists(saturn_path):
        # Load FITS file
        with fits.open(saturn_path) as hdul:
            images['saturn'] = hdul[0].data
            print(f"Loaded Saturn FITS image from {saturn_path}")
    elif os.path.exists(saturn_png_path):
        # Load PNG file
        images['saturn'] = img_as_float(io.imread(saturn_png_path))
        print(f"Loaded Saturn PNG image from {saturn_png_path}")
    else:
        print(f"Saturn image not found. Please place it in {base_dir}")
        return None
    
    # Try to load Saturn PSF
    saturn_psf_path = os.path.join(base_dir, 'saturn_psf.fits')
    saturn_psf_png_path = os.path.join(base_dir, 'saturn_psf.png')
    
    if os.path.exists(saturn_psf_path):
        # Load FITS file
        with fits.open(saturn_psf_path) as hdul:
            images['saturn_psf'] = hdul[0].data
            print(f"Loaded Saturn PSF from {saturn_psf_path}")
    elif os.path.exists(saturn_psf_png_path):
        # Load PNG file
        images['saturn_psf'] = img_as_float(io.imread(saturn_psf_png_path))
        print(f"Loaded Saturn PSF from {saturn_psf_png_path}")
    else:
        print(f"Saturn PSF not found. Please place it in {base_dir}")
        return None
    
    # Ensure PSFs are normalized
    images['jupiter_psf'] = images['jupiter_psf'] / np.sum(images['jupiter_psf'])
    images['saturn_psf'] = images['saturn_psf'] / np.sum(images['saturn_psf'])
    
    return images

def display_astronomical_image(image, title='', cmap='viridis', normalize=True):
    """
    Display an astronomical image with appropriate scaling
    
    Parameters:
    -----------
    image : ndarray
        Image to display
    title : str
        Plot title
    cmap : str
        Colormap name
    normalize : bool
        Whether to apply normalization
    """
    if normalize:
        plt.imshow(image, cmap=cmap, norm=PowerNorm(0.5))
    else:
        plt.imshow(image, cmap=cmap)
    
    plt.title(title)
    plt.colorbar()
    plt.axis('off')

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists('../plots/TP6_Restoration'):
        os.makedirs('../plots/TP6_Restoration')
    
    # Load astronomical images
    images = load_astronomical_images()
    
    if images is None:
        print("Could not load all required images. Please check the messages above.")
        return
    
    # Process Jupiter image
    print("\nProcessing Jupiter image...")
    
    # Apply Wiener filter restoration
    jupiter_restored_wiener = restore_astronomical_image(
        images['jupiter'], 
        images['jupiter_psf'], 
        method='wiener', 
        k=0.005
    )
    
    # Apply inverse filter restoration
    jupiter_restored_inverse = restore_astronomical_image(
        images['jupiter'], 
        images['jupiter_psf'], 
        method='inverse', 
        alpha=0.01
    )
    
    # Process Saturn image
    print("\nProcessing Saturn image...")
    
    # Apply Wiener filter restoration
    saturn_restored_wiener = restore_astronomical_image(
        images['saturn'], 
        images['saturn_psf'], 
        method='wiener', 
        k=0.005
    )
    
    # Apply inverse filter restoration
    saturn_restored_inverse = restore_astronomical_image(
        images['saturn'], 
        images['saturn_psf'], 
        method='inverse', 
        alpha=0.01
    )
    
    # Visualize Jupiter results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    display_astronomical_image(images['jupiter'], title='Original Jupiter')
    
    plt.subplot(132)
    display_astronomical_image(jupiter_restored_inverse, title='Jupiter - Inverse Filter')
    
    plt.subplot(133)
    display_astronomical_image(jupiter_restored_wiener, title='Jupiter - Wiener Filter')
    
    plt.tight_layout()
    plt.savefig('../plots/TP6_Restoration/jupiter_restoration.png', dpi=300)
    plt.show()
    
    # Visualize Jupiter PSF
    plt.figure(figsize=(6, 6))
    display_astronomical_image(images['jupiter_psf'], title='Jupiter PSF')
    plt.tight_layout()
    plt.savefig('../plots/TP6_Restoration/jupiter_psf.png', dpi=300)
    plt.show()
    
    # Visualize Saturn results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    display_astronomical_image(images['saturn'], title='Original Saturn')
    
    plt.subplot(132)
    display_astronomical_image(saturn_restored_inverse, title='Saturn - Inverse Filter')
    
    plt.subplot(133)
    display_astronomical_image(saturn_restored_wiener, title='Saturn - Wiener Filter')
    
    plt.tight_layout()
    plt.savefig('../plots/TP6_Restoration/saturn_restoration.png', dpi=300)
    plt.show()
    
    # Visualize Saturn PSF
    plt.figure(figsize=(6, 6))
    display_astronomical_image(images['saturn_psf'], title='Saturn PSF')
    plt.tight_layout()
    plt.savefig('../plots/TP6_Restoration/saturn_psf.png', dpi=300)
    plt.show()
    
    print("Restoration complete. Results saved to ../plots/TP6_Restoration/")

if __name__ == "__main__":
    main() 