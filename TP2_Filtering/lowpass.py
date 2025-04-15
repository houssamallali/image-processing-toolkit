import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import gaussian
from skimage.filters.rank import mean, median, minimum, maximum
from skimage.morphology import footprint_rectangle
from skimage.util import img_as_ubyte
import os


image = img_as_ubyte(imread('../images/blood.jpg'))

# Filters
mean_3 = mean(image, footprint_rectangle((3, 3)))
mean_25 = mean(image, footprint_rectangle((25, 25)))
med = median(image, footprint_rectangle((5, 5)))
minf = minimum(image, footprint_rectangle((5, 5)))
maxf = maximum(image, footprint_rectangle((5, 5)))
gauss = gaussian(image, sigma=5)

# Plot
plt.figure(figsize=(12, 8))
titles = ['Mean 3x3', 'Mean 25x25', 'Median 5x5', 'Min 5x5', 'Max 5x5', 'Gaussian σ=5']
images = [mean_3, mean_25, med, minf, maxf, gauss]

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(2, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()