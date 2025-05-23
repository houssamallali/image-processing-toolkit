import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
import sys

def contrast_stretching(I, E):
    """
    Apply contrast stretching to an image using the formula s = T(r) = 1/(1+(m/r)^E)
    where m is the mean gray value of the image.
    
    Parameters:
    -----------
    I : ndarray
        Input image (float values between 0 and 1)
    E : float
        Parameter controlling the stretching effect
        
    Returns:
    --------
    stretched : ndarray
        Output image after contrast stretching
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

# Load the osteoblast image
image = img_as_float(imread('../images/osteoblaste.jpg', as_gray=True))

# Apply contrast stretching with different E values
E_values = [10, 20, 1000]
stretched_images = [contrast_stretching(image, E) for E in E_values]

# Plot the results
plt.figure(figsize=(12, 8))

# Original image
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Contrast stretched images
for i, (img, E) in enumerate(zip(stretched_images, E_values)):
    plt.subplot(2, 2, i+2)
    plt.imshow(img, cmap='gray')
    plt.title(f'E = {E}')
    plt.axis('off')

plt.tight_layout()
plt.show() 