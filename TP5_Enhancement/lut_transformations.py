import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.exposure import adjust_gamma
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

# This script demonstrates different LUT (Look-Up Table) transformations
# including gamma correction and contrast stretching

# Create transformation functions
def create_gamma_lut(gamma, size=256):
    """Create a gamma correction LUT."""
    x = np.linspace(0, 1, size)
    y = np.power(x, gamma)
    return x, y

def create_contrast_stretch_lut(m, E, size=256):
    """Create a contrast stretching LUT using s = 1/(1+(m/r)^E)."""
    x = np.linspace(0.001, 1, size)  # Avoid division by zero
    y = 1 / (1 + (m/x)**E)
    return x, y

# Load the image
try:
    # Try to load the Phobos image
    image = img_as_float(imread('../images/phobos.png', as_gray=True))
    image_name = 'Phobos'
except:
    # Fallback to the osteoblast image
    image = img_as_float(imread('../images/osteoblaste.jpg', as_gray=True))
    image_name = 'Osteoblast'

# Create LUTs for visualization
gamma_values = [0.1, 0.5, 1.0, 2.0, 5.0, 25.0]
gamma_luts = [create_gamma_lut(g) for g in gamma_values]

# Create contrast stretch LUTs
m = 0.5  # Mean value (example, would typically use the actual image mean)
E_values = [1, 5, 20, 1000]
contrast_luts = [create_contrast_stretch_lut(m, E) for E in E_values]

# Plot the LUTs
plt.figure(figsize=(15, 10))

# Plot gamma LUTs
plt.subplot(2, 2, 1)
for i, (x, y) in enumerate(gamma_luts):
    plt.plot(x, y, label=f'γ = {gamma_values[i]}')
plt.title('Gamma Correction LUTs')
plt.xlabel('Input Intensity (r)')
plt.ylabel('Output Intensity (s)')
plt.legend()
plt.grid(True)

# Plot contrast stretch LUTs
plt.subplot(2, 2, 2)
for i, (x, y) in enumerate(contrast_luts):
    plt.plot(x, y, label=f'E = {E_values[i]}')
plt.title('Contrast Stretching LUTs')
plt.xlabel('Input Intensity (r)')
plt.ylabel('Output Intensity (s)')
plt.legend()
plt.grid(True)

# Apply transformations and display
# Gamma correction
plt.subplot(2, 2, 3)
gamma_corrected = adjust_gamma(image, gamma=0.5)
plt.imshow(gamma_corrected, cmap='gray')
plt.title(f'{image_name} with γ = 0.5')
plt.axis('off')

# Contrast stretching
plt.subplot(2, 2, 4)
# Using our locally defined function
contrast_stretched = contrast_stretching(image, E=20)
plt.imshow(contrast_stretched, cmap='gray')
plt.title(f'{image_name} with Contrast Stretching (E = 20)')
plt.axis('off')

plt.tight_layout()
plt.show() 