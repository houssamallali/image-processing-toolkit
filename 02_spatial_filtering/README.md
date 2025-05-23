# TP2: Spatial Filtering

Professional implementation of spatial filtering techniques for image enhancement, noise reduction, and feature extraction.

## Overview

Spatial filtering is a fundamental image processing technique that operates directly on pixel neighborhoods using convolution with various kernels. This module provides comprehensive implementations of different filtering approaches with detailed analysis and visualization.

## Physics and Theory

### Convolution Operation
Convolution is a mathematical operation that combines an image with a kernel (small matrix) to produce filtered output:

```
g(x,y) = Σ Σ f(i,j) * h(x-i, y-j)
```

Where:
- `f(i,j)` is the input image
- `h(x,y)` is the convolution kernel
- `g(x,y)` is the filtered output

### Filter Types

**Low-Pass Filters:**
- Remove high-frequency components (noise, fine details)
- Preserve low-frequency components (main structures)
- Examples: Mean, Gaussian, Median filters

**High-Pass Filters:**
- Enhance high-frequency components (edges, details)
- Suppress low-frequency components
- Examples: Laplacian, Sobel, Unsharp masking

**Band-Pass Filters:**
- Preserve specific frequency ranges
- Combine low-pass and high-pass characteristics

## Key Concepts

- **Spatial Filtering**: Operations performed directly on pixel neighborhoods
- **Linear Filters**: Output is a weighted sum of input pixels (e.g., mean, Gaussian)
- **Non-linear Filters**: Output doesn't follow linear combination (e.g., median, min, max)
- **Convolution**: Mathematical operation combining image with a kernel
- **Kernel Properties**: Size, normalization, and symmetry affect filter behavior

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