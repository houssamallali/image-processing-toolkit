# TP1: Basic Image Processing

This module covers fundamental image processing operations and concepts.

## Key Concepts

- **Image Representation**: Digital images are represented as arrays where each element corresponds to a pixel value.
- **Color Channels**: Color images typically consist of three channels (Red, Green, Blue).
- **Image I/O**: Loading and saving images in various formats (JPG, PNG, etc.).
- **Image Visualization**: Displaying images and their properties.

## Files

- `firsttest.py`: Demonstrates basic image loading, RGB channel separation, visualization, and saving with different compression levels.

## Core Operations

### Loading Images
```python
from skimage.io import imread
image = imread('../images/retina.jpg')
```

### Image Properties
- Shape: Dimensions of the image (height, width, channels)
- Dtype: Data type of pixel values (uint8, float, etc.)

### Channel Separation
```python
red = image[:, :, 0]    # Red channel
green = image[:, :, 1]  # Green channel
blue = image[:, :, 2]   # Blue channel
```

### Image Compression
Different quality settings affect file size and visual quality when saving images:
```python
from skimage.io import imsave
imsave("output.jpg", image, quality=50)  # Lower quality = smaller file size
```

## Practical Applications

- Medical imaging visualization
- Simple image analysis
- Understanding digital image structure
- File format conversions and optimization

## Additional Resources

- [scikit-image Documentation](https://scikit-image.org/docs/stable/)
- [Matplotlib Image Tutorial](https://matplotlib.org/stable/tutorials/introductory/images.html)
- [Python Pillow Library](https://pillow.readthedocs.io/en/stable/) 