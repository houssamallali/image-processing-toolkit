import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, filters
from scipy import signal
import os
from matplotlib.colors import PowerNorm

def create_jupiter_image(size=256):
    """
    Create a synthetic Jupiter image
    
    Parameters:
    -----------
    size : int
        Size of the image (width and height)
        
    Returns:
    --------
    ndarray
        Synthetic Jupiter image
    """
    # Create base image
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    r = np.sqrt(x**2 + y**2)
    
    # Create Jupiter disk with limb darkening
    jupiter = np.zeros((size, size))
    disk_mask = r <= 0.8
    jupiter[disk_mask] = 0.8 * (1 - 0.3 * r[disk_mask]**2)
    
    # Add bands/stripes
    for i in range(10):
        center = 0.6 * (i/10 - 0.45)
        width = 0.03 + 0.02 * np.sin(i)
        intensity = 0.2 + 0.1 * np.sin(i)
        band = np.exp(-(y - center)**2 / (2 * width**2))
        jupiter[disk_mask] += intensity * band[disk_mask] * (i % 2 * 2 - 1)
    
    # Add some features/storms
    storm_x, storm_y = -0.3, -0.1
    storm_r = np.sqrt((x - storm_x)**2 + (y - storm_y)**2)
    storm_mask = storm_r < 0.1
    jupiter[storm_mask] = 0.95
    
    # Normalize to [0, 1]
    jupiter = (jupiter - np.min(jupiter)) / (np.max(jupiter) - np.min(jupiter))
    
    # Make background black
    jupiter[~disk_mask] = 0
    
    return jupiter

def create_saturn_image(size=256):
    """
    Create a synthetic Saturn image with rings
    
    Parameters:
    -----------
    size : int
        Size of the image (width and height)
        
    Returns:
    --------
    ndarray
        Synthetic Saturn image
    """
    # Create base image
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    r = np.sqrt(x**2 + y**2)
    
    # Create Saturn disk with limb darkening
    saturn = np.zeros((size, size))
    disk_mask = (x/0.6)**2 + (y/0.5)**2 <= 1
    saturn[disk_mask] = 0.7 * (1 - 0.3 * ((x[disk_mask]/0.6)**2 + (y[disk_mask]/0.5)**2))
    
    # Add bands/stripes (subtle for Saturn)
    for i in range(5):
        center = 0.3 * (i/5 - 0.4)
        width = 0.05
        intensity = 0.1
        band = np.exp(-(y - center)**2 / (2 * width**2))
        saturn[disk_mask] += intensity * band[disk_mask] * (i % 2 * 2 - 1)
    
    # Create rings
    # Inner ring
    inner_ring_mask = (((x/0.9)**2 + (y/0.2)**2 <= 1) & 
                      ((x/0.7)**2 + (y/0.15)**2 > 1) &
                      ~disk_mask)
    saturn[inner_ring_mask] = 0.5
    
    # Middle ring
    middle_ring_mask = (((x/1.1)**2 + (y/0.25)**2 <= 1) & 
                       ((x/0.9)**2 + (y/0.2)**2 > 1) &
                       ~disk_mask)
    saturn[middle_ring_mask] = 0.8
    
    # Outer ring
    outer_ring_mask = (((x/1.3)**2 + (y/0.3)**2 <= 1) & 
                      ((x/1.1)**2 + (y/0.25)**2 > 1) &
                      ~disk_mask)
    saturn[outer_ring_mask] = 0.4
    
    # Add Cassini Division (gap in rings)
    cassini_mask = (((x/1.0)**2 + (y/0.23)**2 <= 1) & 
                  ((x/0.95)**2 + (y/0.21)**2 > 1) &
                  ~disk_mask)
    saturn[cassini_mask] = 0.1
    
    # Add ring shadow on planet
    shadow_mask = disk_mask & (y > 0.1)
    shadow_strength = np.clip((y[shadow_mask] - 0.1) / 0.4, 0, 1)
    saturn[shadow_mask] *= 1 - 0.5 * shadow_strength
    
    # Normalize to [0, 1]
    saturn = (saturn - np.min(saturn)) / (np.max(saturn) - np.min(saturn))
    
    return saturn

def create_point_spread_function(size=64, sigma=2):
    """
    Create a point spread function for astronomical imaging
    
    Parameters:
    -----------
    size : int
        Size of the PSF (width and height)
    sigma : float
        Width of the PSF (Gaussian sigma)
        
    Returns:
    --------
    ndarray
        PSF image
    """
    x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))
    r = np.sqrt(x**2 + y**2)
    
    # Create Gaussian PSF
    psf = np.exp(-r**2 / (2 * sigma**2))
    
    # Add diffraction spikes (common in telescope images)
    spike_x = 0.3 * np.exp(-np.abs(y)**2 / 0.05)
    spike_y = 0.3 * np.exp(-np.abs(x)**2 / 0.05)
    
    psf += spike_x + spike_y
    
    # Normalize to sum to 1
    psf = psf / np.sum(psf)
    
    return psf

def create_circular_psf(size=64, radius=15):
    """
    Create a circular PSF with diffraction pattern
    
    Parameters:
    -----------
    size : int
        Size of the PSF (width and height)
    radius : int
        Radius of the circular aperture
        
    Returns:
    --------
    ndarray
        PSF image
    """
    # Create meshgrid
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    r = np.sqrt(x*x + y*y)
    
    # Create circular aperture
    aperture = np.zeros((size, size))
    aperture[r <= radius] = 1
    
    # Compute PSF as squared magnitude of Fourier transform of aperture
    # (simplified model of diffraction)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(aperture)))**2
    
    # Normalize
    psf = psf / np.sum(psf)
    
    return psf

def apply_psf_to_image(image, psf):
    """
    Apply a PSF to an image through convolution
    
    Parameters:
    -----------
    image : ndarray
        Input image
    psf : ndarray
        Point Spread Function
        
    Returns:
    --------
    ndarray
        Blurred image
    """
    # Pad PSF to match image size if needed
    if psf.shape[0] < image.shape[0] or psf.shape[1] < image.shape[1]:
        pad_width = ((0, max(0, image.shape[0] - psf.shape[0])), 
                     (0, max(0, image.shape[1] - psf.shape[1])))
        psf_padded = np.pad(psf, pad_width, mode='constant')
        # Center the PSF
        psf_padded = np.roll(psf_padded, shift=(-psf.shape[0]//2, -psf.shape[1]//2), axis=(0, 1))
    else:
        psf_padded = psf
    
    # Apply convolution
    # Using FFT-based convolution for efficiency
    blurred = signal.fftconvolve(image, psf_padded, mode='same')
    
    return blurred

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
    # Create output directory
    base_dir = '../images/astronomy'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create output plots directory
    if not os.path.exists('../plots/TP6_Restoration'):
        os.makedirs('../plots/TP6_Restoration')
    
    # Create Jupiter image
    print("Creating synthetic Jupiter image...")
    jupiter = create_jupiter_image(size=256)
    
    # Create Jupiter PSF
    jupiter_psf = create_point_spread_function(size=64, sigma=1.5)
    
    # Create Saturn image
    print("Creating synthetic Saturn image...")
    saturn = create_saturn_image(size=256)
    
    # Create Saturn PSF
    saturn_psf = create_circular_psf(size=64, radius=10)
    
    # Save images
    plt.figure(figsize=(8, 8))
    display_astronomical_image(jupiter, title='Synthetic Jupiter Image')
    plt.tight_layout()
    plt.savefig(f'{base_dir}/jupiter.png', dpi=300)
    
    plt.figure(figsize=(8, 8))
    display_astronomical_image(jupiter_psf, title='Jupiter PSF')
    plt.tight_layout()
    plt.savefig(f'{base_dir}/jupiter_psf.png', dpi=300)
    
    plt.figure(figsize=(8, 8))
    display_astronomical_image(saturn, title='Synthetic Saturn Image')
    plt.tight_layout()
    plt.savefig(f'{base_dir}/saturn.png', dpi=300)
    
    plt.figure(figsize=(8, 8))
    display_astronomical_image(saturn_psf, title='Saturn PSF')
    plt.tight_layout()
    plt.savefig(f'{base_dir}/saturn_psf.png', dpi=300)
    
    # Save NumPy arrays for restoration
    np.save(f'{base_dir}/jupiter.npy', jupiter)
    np.save(f'{base_dir}/jupiter_psf.npy', jupiter_psf)
    np.save(f'{base_dir}/saturn.npy', saturn)
    np.save(f'{base_dir}/saturn_psf.npy', saturn_psf)
    
    # Create example of blurred images
    jupiter_blurred = apply_psf_to_image(jupiter, jupiter_psf)
    saturn_blurred = apply_psf_to_image(saturn, saturn_psf)
    
    # Display and save comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    display_astronomical_image(jupiter, title='Original Jupiter')
    
    plt.subplot(132)
    display_astronomical_image(jupiter_psf, title='Jupiter PSF')
    
    plt.subplot(133)
    display_astronomical_image(jupiter_blurred, title='Blurred Jupiter')
    
    plt.tight_layout()
    plt.savefig('../plots/TP6_Restoration/jupiter_blurring_example.png', dpi=300)
    plt.show()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    display_astronomical_image(saturn, title='Original Saturn')
    
    plt.subplot(132)
    display_astronomical_image(saturn_psf, title='Saturn PSF')
    
    plt.subplot(133)
    display_astronomical_image(saturn_blurred, title='Blurred Saturn')
    
    plt.tight_layout()
    plt.savefig('../plots/TP6_Restoration/saturn_blurring_example.png', dpi=300)
    plt.show()
    
    print(f"Created and saved synthetic astronomy images to {base_dir}/")
    print("Created and saved example plots to ../plots/TP6_Restoration/")

if __name__ == "__main__":
    main() 