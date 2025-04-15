import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import gaussian, laplace
from skimage.filters.rank import mean
from skimage.morphology import footprint_rectangle
from skimage.util import img_as_float

image = img_as_float(imread('blood.jpg'))


mean_25 = mean(image, footprint_rectangle((5, 5)))
gauss = gaussian(image, sigma=2)
laplacien = laplace(image)

# High-pass = Original - Low-pass
hp_mean = image - mean_25
hp_gauss = image - gauss



# Plot
plt.figure(figsize=(10, 5))
titles = ['High-Pass (Mean 5x5)', 'High-Pass (Gaussian σ=2)','le vrai']
images = [hp_mean, hp_gauss,laplacien]

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()