import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.filters import gaussian
import numpy as np


cornea = img_as_float(imread('../images/cornee.png', as_gray=True))

# La transforme de fourier
fft = np.fft.fft2(cornea)
fft_shift = np.fft.fftshift(fft)
amplitude = np.abs(fft_shift)

# On applique le filtre guassien
blurred = gaussian(amplitude, sigma=5)



' Plot twist '

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.log(1 + blurred), cmap='gray')
plt.title('Filtered Amplitude')
plt.axis('off')

plt.subplot(1, 2, 2)
midY = blurred.shape[0] // 2
plt.plot(np.log(1 + blurred[midY, :]))
plt.title('Peak observation and cell frequency')

plt.tight_layout()
plt.show()
