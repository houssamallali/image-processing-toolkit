import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
import numpy as np
import os

image = img_as_float(imread('../images/cornee.png', as_gray=True))
fft = np.fft.fft2(image)
fft_shift = np.fft.fftshift(fft)

amplitude = np.abs(fft_shift)
phase = np.angle(fft_shift)

# Transformée de fourier inverse
recon_full = np.fft.ifft2(np.fft.ifftshift(fft_shift)).real

# 2. Inverse  amplitude
amp_only = amplitude * np.exp(1j * 0)
recon_amp = np.fft.ifft2(np.fft.ifftshift(amp_only)).real

# 3. Inverse  phase (m = 1)
complex_phase = np.exp(1j * phase)
recon_phase = np.fft.ifft2(np.fft.ifftshift(complex_phase)).real

""" Pour l'affichage rapide"""

titles = ['Full Inverse FT', 'Inverse FT on Amplitude', 'Inverse FT on Phase']
images = [recon_full, recon_amp, recon_phase]

plt.figure(figsize=(10, 10))
for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
