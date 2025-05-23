import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def motion_psf(length=15, angle=45, size=32):
    """
    Generate a motion blur PSF
    
    Parameters:
    -----------
    length : int
        Length of the motion blur
    angle : float
        Angle of the motion blur in degrees
    size : int
        Size of the PSF array
        
    Returns:
    --------
    ndarray
        Motion blur PSF
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Create PSF kernel
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

def save_psf_data(psf, filename):
    """
    Save PSF data to a .npy file
    
    Parameters:
    -----------
    psf : ndarray
        PSF to save
    filename : str
        Output filename
    """
    np.save(filename, psf)
    print(f"PSF saved to {filename}")

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists('../plots/TP6_Restoration'):
        os.makedirs('../plots/TP6_Restoration')
    
    # Create different motion PSFs
    print("Creating motion PSFs...")
    
    # Linear motion blur
    linear_psf = motion_psf(length=15, angle=45, size=32)
    
    # Horizontal motion blur
    horizontal_psf = motion_psf(length=15, angle=0, size=32)
    
    # Vertical motion blur
    vertical_psf = motion_psf(length=15, angle=90, size=32)
    
    # Circular motion blur (approximated by multiple motion blurs at different angles)
    circular_psf = np.zeros((32, 32))
    for angle in range(0, 360, 30):  # Angles in 30-degree increments
        circular_psf += motion_psf(length=10, angle=angle, size=32)
    circular_psf /= circular_psf.sum()  # Renormalize
    
    # Save PSFs
    save_psf_data(linear_psf, '../plots/TP6_Restoration/psf_motion.npy')
    save_psf_data(horizontal_psf, '../plots/TP6_Restoration/psf_horizontal.npy')
    save_psf_data(vertical_psf, '../plots/TP6_Restoration/psf_vertical.npy')
    save_psf_data(circular_psf, '../plots/TP6_Restoration/psf_circular.npy')
    
    # Visualize PSFs
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.imshow(linear_psf, cmap='viridis')
    plt.title('Linear Motion PSF (45°)')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(horizontal_psf, cmap='viridis')
    plt.title('Horizontal Motion PSF')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(vertical_psf, cmap='viridis')
    plt.title('Vertical Motion PSF')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(circular_psf, cmap='viridis')
    plt.title('Circular Motion PSF')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../plots/TP6_Restoration/motion_psfs.png', dpi=300)
    plt.show()
    
    # Create sample blurred images
    print("Creating sample blurred images...")
    
    # Create checkerboard pattern
    checkerboard = np.kron(([1, 0] * 4, [0, 1] * 4) * 4, np.ones((8, 8)))
    
    # Apply different motion blurs
    blurred_linear = signal.convolve2d(checkerboard, linear_psf, mode='same', boundary='wrap')
    blurred_horizontal = signal.convolve2d(checkerboard, horizontal_psf, mode='same', boundary='wrap')
    blurred_vertical = signal.convolve2d(checkerboard, vertical_psf, mode='same', boundary='wrap')
    blurred_circular = signal.convolve2d(checkerboard, circular_psf, mode='same', boundary='wrap')
    
    # Visualize blurred images
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.imshow(checkerboard, cmap='gray')
    plt.title('Original Checkerboard')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(blurred_linear, cmap='gray')
    plt.title('Linear Motion Blur (45°)')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(blurred_horizontal, cmap='gray')
    plt.title('Horizontal Motion Blur')
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(blurred_vertical, cmap='gray')
    plt.title('Vertical Motion Blur')
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(blurred_circular, cmap='gray')
    plt.title('Circular Motion Blur')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../plots/TP6_Restoration/blurred_checkerboards.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main() 