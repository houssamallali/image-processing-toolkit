# TP4: Image Segmentation

This module covers image segmentation techniques, which partition an image into multiple segments to simplify representation and make analysis easier.

## Key Concepts

- **Image Segmentation**: The process of dividing an image into multiple meaningful regions.
- **Thresholding**: Classifying pixels based on intensity values.
- **Clustering**: Grouping pixels with similar characteristics.
- **Region Growing**: Expanding regions from seed points based on similarity criteria.
- **Edge Detection**: Identifying boundaries between regions.
- **Watershed Algorithm**: A region-based segmentation approach inspired by geological watersheds.

## Files

- `thresholding.py`: Demonstrates basic manual thresholding for binary segmentation.
- `kmeans_segmentation.py`: Implements K-means clustering for automatic image segmentation.
- `otsu_segmentation.py`: Shows Otsu's method for automatic threshold selection.
- `advanced_segmentation.py`: Contains advanced segmentation techniques like region-based and graph-based methods.

## Core Operations

### Manual Thresholding
```python
threshold = 100  # manually selected value
binary_image = image > threshold
```

### K-means Segmentation
```python
# Iterative threshold calculation (k-means with k=2)
T = 0.5 * (image.max() + image.min())  # initial threshold
epsilon = 1e-4
done = False

while not done:
    G1 = image[image >= T]  # pixels above threshold
    G2 = image[image < T]   # pixels below threshold
    
    mu1 = G1.mean() if len(G1) > 0 else 0
    mu2 = G2.mean() if len(G2) > 0 else 0
    
    T_new = 0.5 * (mu1 + mu2)
    
    if abs(T_new - T) < epsilon:
        done = True
    T = T_new

segmented = image > T
```

### Otsu's Method
```python
from skimage.filters import threshold_otsu
otsu_threshold = threshold_otsu(image)
binary_image = image > otsu_threshold
```

## Mathematical Foundation

### Otsu's Method
Otsu's method maximizes the between-class variance:

```
σ²_b(t) = w₀(t)w₁(t)[μ₀(t) - μ₁(t)]²
```

Where:
- w₀, w₁ are the probabilities of the two classes
- μ₀, μ₁ are the mean values of the two classes
- t is the threshold value

### K-means Algorithm
The K-means algorithm aims to minimize the within-cluster sum of squares:

```
J = Σ(i=1 to k) Σ(x in Sᵢ) ||x - μᵢ||²
```

Where:
- k is the number of clusters
- Sᵢ is the set of points in cluster i
- μᵢ is the mean of points in cluster i

## Practical Applications

- **Medical Image Analysis**: Identifying organs, tumors, and anatomical structures.
- **Object Detection**: Isolating objects of interest from backgrounds.
- **Cell Counting**: Quantifying cells in microscopy images.
- **Document Processing**: Separating text from background in scanned documents.
- **Remote Sensing**: Identifying land use patterns from satellite imagery.
- **Quality Control**: Detecting defects in industrial products.

## Additional Resources

- [Interactive Image Segmentation Tutorial](https://scikit-image.org/docs/stable/auto_examples/applications/plot_coins_segmentation.html)
- [Review of Image Segmentation Techniques](https://www.sciencedirect.com/science/article/pii/S2090447914000355)
- [Thresholding Techniques Overview](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html) 