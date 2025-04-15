import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.signal import convolve2d
import numpy as np



image = img_as_float(imread('../images/blood.jpg'))
#kernels
mean_kernel = (1/9) * np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

laplacian_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

gaussian_kernel = (1/16) * np.array([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]])


conv_mean = convolve2d(image, mean_kernel, mode='same', boundary='symm')
conv_laplacian = convolve2d(image, laplacian_kernel, mode='same', boundary='symm')
conv_gaussian = convolve2d(image, gaussian_kernel, mode='same', boundary='symm')

# Plot results
plt.figure(figsize=(10, 10))
titles = ['Mean Filter', 'Laplacian Filter', 'Gaussian Filter']
images = [conv_mean, conv_laplacian, conv_gaussian]

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()