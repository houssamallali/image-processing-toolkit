import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
import numpy as np

image = img_as_float(imread('../images/cornee.png', as_gray=True))
fft = np.fft.fft2(image)
fft_shift = np.fft.fftshift(fft)

def LowPassFilter(spectrum, cut):
    X, Y = spectrum.shape
    mask = np.zeros((X, Y), "int")
    mx, my = X//2, Y//2
    mask[mx-cut:mx+cut, my-cut:my+cut] = 1
    return spectrum * mask

def HighPassFilter(spectrum, cut):
    X, Y = spectrum.shape
    mask = np.ones((X, Y), "int")
    mx, my = X//2, Y//2
    mask[mx-cut:mx+cut, my-cut:my+cut] = 0
    return spectrum * mask

# filters
low = LowPassFilter(fft_shift, 30)
high = HighPassFilter(fft_shift, 30)

# Inverse FFT
img_low = np.fft.ifft2(np.fft.ifftshift(low)).real
img_high = np.fft.ifft2(np.fft.ifftshift(high)).real


" Les plot twist"

images = [img_low, img_high]
titles = ['Reconstruction after LP filtering', 'Reconstruction after HP filtering']

plt.figure(figsize=(10, 10))
for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
