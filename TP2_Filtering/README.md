# TP2: Image Filtering Techniques

This module covers various filtering techniques used for image enhancement, noise reduction, and feature extraction.

## Key Concepts

- **Spatial Filtering**: Processing images in the spatial domain using convolution with kernels.
- **Linear Filters**: Filters where each output pixel is a weighted sum of input pixels (e.g., mean, Gaussian).
- **Non-linear Filters**: Filters that don't use weighted sums (e.g., median, min, max).
- **Convolution**: The mathematical operation of applying a kernel to an image.
- **Aliasing**: Visual artifacts that occur when sampling frequency is insufficient.

## Files

- `lowpass.py`: Demonstrates low-pass filtering techniques (mean, median, Gaussian) that reduce high frequencies (details and noise).
- `highpass.py`: Implements high-pass filtering to enhance edges and details by removing low frequencies.
- `convolution.py`: Shows custom convolution operations with different kernels.
- `enhancement.py`: Contains advanced image enhancement techniques.
- `aliasing_effect.py`: Illustrates aliasing artifacts and how to avoid them.

## Core Operations

### Low-Pass Filtering
```python
# Mean filter (averaging)
from skimage.filters.rank import mean
from skimage.morphology import footprint_rectangle
mean_filtered = mean(image, footprint_rectangle((3, 3)))

# Gaussian filter (weighted average with normal distribution)
from skimage.filters import gaussian
gaussian_filtered = gaussian(image, sigma=1.5)
```

### High-Pass Filtering
```python
# Using Laplacian kernel
laplacian_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])
from scipy.signal import convolve2d
high_pass = convolve2d(image, laplacian_kernel, mode='same', boundary='symm')
```

### Custom Convolution
```python
# Custom kernel for edge detection (Sobel)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
edges_x = convolve2d(image, sobel_x, mode='same', boundary='symm')
```

## Practical Applications

- **Noise Reduction**: Using low-pass filters to remove noise from images.
- **Edge Detection**: Using high-pass filters to highlight boundaries.
- **Image Sharpening**: Enhancing details and edges for better visibility.
- **Blur Removal**: Removing unwanted blur from images.
- **Feature Extraction**: Identifying specific features in medical or scientific images.

## Additional Resources

- [Convolution Explanation](https://www.cs.toronto.edu/~jepson/csc320/notes/linearFilters.pdf)
- [Image Filtering Tutorial](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html)
- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html) 