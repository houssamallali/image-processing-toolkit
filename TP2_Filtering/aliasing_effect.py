import numpy as np
import matplotlib.pyplot as plt


f0 = 30
fs = 100

t = np.arange(0, 1, 1/fs)
xx, yy = np.meshgrid(t, t)

r = np.sqrt(xx**2 + yy**2)
g = np.sin(2 * np.pi * f0 * r)

# Plot
plt.figure(figsize=(5, 5))
plt.imshow(g, cmap='gray', extent=(0, 1, 0, 1))
plt.title(f'Aliasing Example\nf0={f0}, fs={fs}')
plt.axis('off')
plt.tight_layout()
plt.show()