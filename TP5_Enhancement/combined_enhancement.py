import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.exposure import adjust_gamma, rescale_intensity
import sys

# Define contrast_stretching locally since we can't import the module directly
def contrast_stretching(I, E):
    """
    Apply contrast stretching to an image using the formula s = T(r) = 1/(1+(m/r)^E)
    where m is the mean gray value of the image.
    """
    # Get smallest float value to avoid division by zero
    epsilon = sys.float_info.epsilon
    
    # Calculate mean value of the image
    m = np.mean(I)
    
    # Ensure image is of float type
    I = I.astype('float')
    
    # Apply contrast stretching formula
    stretched = 1. / (1. + (m/(I + epsilon))**E)
    
    return stretched

# Load the Phobos Mars moon image
# If not available, use the osteoblaste image instead
try:
    image = img_as_float(imread('../images/phobos.png', as_gray=True))
except:
    image = img_as_float(imread('../images/osteoblaste.jpg', as_gray=True))

# Create a pipeline of enhancement operations
def enhance_image(image, gamma=1.0, contrast_E=10):
    """
    Apply a sequence of enhancement operations to an image.
    
    Parameters:
    -----------
    image : ndarray
        Input image (float values between 0 and 1)
    gamma : float
        Gamma correction value
    contrast_E : float
        Parameter for contrast stretching
        
    Returns:
    --------
    enhanced : ndarray
        Enhanced image
    """
    # Step 1: Apply gamma correction
    gamma_corrected = adjust_gamma(image, gamma)
    
    # Step 2: Apply contrast stretching
    contrast_stretched = contrast_stretching(gamma_corrected, contrast_E)
    
    # Step 3: Rescale intensity to full range
    enhanced = rescale_intensity(contrast_stretched)
    
    return enhanced, gamma_corrected, contrast_stretched

# Create different enhancement combinations
enhancements = [
    enhance_image(image, gamma=0.5, contrast_E=5),
    enhance_image(image, gamma=1.5, contrast_E=15),
    enhance_image(image, gamma=2.0, contrast_E=10)
]

# Plot results
plt.figure(figsize=(15, 12))

# Original image
plt.subplot(4, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Processed images
titles = [
    ['Gamma=0.5', 'Contrast E=5', 'Final Enhanced'],
    ['Gamma=1.5', 'Contrast E=15', 'Final Enhanced'],
    ['Gamma=2.0', 'Contrast E=10', 'Final Enhanced']
]

for i, (enhanced_set, title_set) in enumerate(zip(enhancements, titles)):
    final, gamma_stage, contrast_stage = enhanced_set
    
    # Show gamma correction stage
    plt.subplot(4, 3, i*3+4)
    plt.imshow(gamma_stage, cmap='gray')
    plt.title(f'Stage 1: {title_set[0]}')
    plt.axis('off')
    
    # Show contrast stretching stage
    plt.subplot(4, 3, i*3+5)
    plt.imshow(contrast_stage, cmap='gray')
    plt.title(f'Stage 2: {title_set[1]}')
    plt.axis('off')
    
    # Show final enhanced image
    plt.subplot(4, 3, i*3+6)
    plt.imshow(final, cmap='gray')
    plt.title(f'Final: {title_set[2]}')
    plt.axis('off')

plt.tight_layout()
plt.show() 