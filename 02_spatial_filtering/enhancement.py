import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float, img_as_ubyte
from scipy.signal import convolve2d
import numpy as np

# Load and normalize image
image = img_as_float(imread('../images/osteoblaste.jpg', as_gray=True))

# Laplacian kernel
laplacian_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

# High-pass using Laplacian
hp = convolve2d(image, laplacian_kernel, mode='same', boundary='symm')

# E(f) = f + HP(f)
enhanced = image + hp

# Parameterized enhancement
alpha = 1.5
enhanced_param = alpha * image + hp

# final result
enhanced = np.clip(enhanced, 0, 1)
enhanced_param = np.clip(enhanced_param, 0, 1)

# Plot
plt.figure(figsize=(12, 4))
titles = ['Original', 'Enhanced (f + HP)', f'Enhanced (αf + HP), α={alpha}']
images = [image, enhanced, enhanced_param]

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()