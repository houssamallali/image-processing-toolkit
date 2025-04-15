import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_ubyte
import numpy as np
import os

# Update path to image
image = img_as_ubyte(imread('../images/cells.png', as_gray=True))

#histogramme

plt.figure(figsize=(6, 4))
plt.hist(image.ravel(), bins=256, range=(0, 255), color='gray')
plt.title("Histogramme de l'image")
plt.xlabel("Intensité")
plt.ylabel("Nombre de pixels")
plt.tight_layout()
plt.show()

# Seuillage manuel (ex: seuil à 100)
threshold = 100
segmented = image > threshold

# Affichage image originale + segmentée
plt.figure(figsize=(10, 5))
titles = ['Image originale', f'Seuillage manuel (>{threshold})']
images = [image, segmented]

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
