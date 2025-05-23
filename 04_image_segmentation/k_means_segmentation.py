import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
import numpy as np
from skimage.filters import threshold_otsu
import os

image = img_as_float(imread('../images/cells.png', as_gray=True))

T = 0.5 * (image.max() + image.min())
epsilon = 1e-4
done = False

# Une chose très classique ...
while not done:
    G1 = image[image >= T]
    G2 = image[image < T]

    mu1 = G1.mean() if len(G1) > 0 else 0
    mu2 = G2.mean() if len(G2) > 0 else 0

    T_new = 0.5 * (mu1 + mu2)

    if abs(T_new - T) < epsilon:
        done = True
    T = T_new

segmented = image > T

plt.figure(figsize=(10, 5))
titles = ['Image originale', f'Seuillage k-means (T={T:.3f})']
images = [image, segmented]

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

def autothresh(image):
    s = 0.5 * (np.min(image) + np.max(image))
    done = False
    while not done:
        B = image > s
        sNext = 0.5 * (np.mean(image[B]) + np.mean(image[~B]))
        done = abs(s - sNext) < 1e-4
        s = sNext
    return s

s_auto = autothresh(image)
s_otsu = threshold_otsu(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image > s_auto, cmap='gray')
plt.title(f'Seuillage automatique (s = {s_auto:.3f})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image > s_otsu, cmap='gray')
plt.title(f'Seuillage Otsu (s = {s_otsu:.3f})')
plt.axis('off')

plt.tight_layout()
plt.show()
