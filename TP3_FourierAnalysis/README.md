# TP3: Fourier Transform Analysis

This module explores frequency domain processing using Fourier transforms, a powerful technique for analyzing and manipulating images in the frequency domain.

## Key Concepts

- **Fourier Transform**: Decomposes an image into its frequency components (sinusoidal basis functions).
- **Frequency Domain**: Representation of an image in terms of frequency rather than spatial coordinates.
- **Amplitude and Phase**: The two components of the Fourier transform.
- **Spatial Frequency**: How rapidly pixel values change across an image.
- **Filtering in Frequency Domain**: Selective attenuation or amplification of frequency components.

## Files

- `2Dfourier.py`: Demonstrates two-dimensional Fourier transforms and visualization.
- `inversefourier.py`: Shows how to reconstruct images from frequency components.
- `LP_and_HP_filtering.py`: Implements low-pass and high-pass filtering in the frequency domain.
- `application.py`: Illustrates practical applications of Fourier transforms in image processing.

## Core Operations

### Forward Fourier Transform
```python
import numpy as np
# Calculate 2D FFT
fft = np.fft.fft2(image)
# Shift zero frequency to center
fft_shifted = np.fft.fftshift(fft)
# Visualize amplitude spectrum (log scale)
amplitude_spectrum = np.log(1 + np.abs(fft_shifted))
```

### Inverse Fourier Transform
```python
# Inverse shift
inverse_shift = np.fft.ifftshift(fft_shifted)
# Inverse FFT to reconstruct image
reconstructed_image = np.fft.ifft2(inverse_shift).real
```

### Frequency Domain Filtering
```python
# Create frequency domain filter (e.g., low-pass)
rows, cols = image.shape
crow, ccol = rows//2, cols//2
mask = np.ones((rows, cols), np.uint8)
r = 30  # radius
mask[crow-r:crow+r, ccol-r:ccol+r] = 0  # Remove high frequencies

# Apply filter
fft_filtered = fft_shifted * mask
# Inverse transform to get filtered image
filtered_image = np.fft.ifft2(np.fft.ifftshift(fft_filtered)).real
```

## Mathematical Background

The Discrete Fourier Transform (DFT) of an image f(x,y) of size M×N is given by:

```
F(u,v) = (1/MN) * Σ(x=0 to M-1) Σ(y=0 to N-1) f(x,y) * exp[-j2π(ux/M + vy/N)]
```

The inverse DFT is:

```
f(x,y) = Σ(u=0 to M-1) Σ(v=0 to N-1) F(u,v) * exp[j2π(ux/M + vy/N)]
```

## Practical Applications

- **Image Compression**: Removing high frequencies to reduce data size.
- **Noise Removal**: Filtering specific frequency components to eliminate noise.
- **Feature Detection**: Identifying periodic patterns and structures.
- **Image Enhancement**: Selectively amplifying frequency components.
- **Medical Imaging**: MRI and CT scan processing.
- **Texture Analysis**: Analyzing repeating patterns in surfaces.

## Additional Resources

- [Fourier Transform: A Practical Introduction](https://www.cs.unm.edu/~brayer/vision/fourier.html)
- [Image Processing in Frequency Domain](https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm)
- [NumPy FFT Documentation](https://numpy.org/doc/stable/reference/routines.fft.html) 