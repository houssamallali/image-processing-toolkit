import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
import numpy as np

image = img_as_float(imread('../images/cornee.png', as_gray=True))

fft = np.fft.fft2(image)
fft_shift = np.fft.fftshift(fft)

amplitude = np.log(1 + np.abs(fft_shift))
phase = np.angle(fft_shift)





plt.figure(figsize=(10, 10))
titles = ['Original Image', 'Amplitude Spectrum', 'Phase Spectrum']
images = [image, amplitude, phase]

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
