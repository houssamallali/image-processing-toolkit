# TP3 : Analyse de Fourier

Ce module traite l'image dans le domaine fréquentiel grâce à la transformée de Fourier.

## Points importants

- **Transformée de Fourier** : décompose l'image en composantes de fréquences.
- **Domaine fréquentiel** : représentation en fréquence plutôt qu'en pixels.
- **Amplitude et phase** : informations essentielles de la TFD.
La TFD transforme la fonction image en somme de sinusoides de differentes frequences. Les hautes frequences traduisent des variations rapides tandis que les basses frequences decrivent les structures globales. Manipuler ce spectre permet de filtrer ou de compresser l'information plus efficacement que dans le domaine spatial.


## Scripts

- `2Dfourier.py` : calcul de la transformée 2D et visualisation du spectre.
- `inversefourier.py` : reconstruction de l'image par transformée inverse.
- `LP_and_HP_filtering.py` : filtrage passe-bas et passe-haut en fréquence.
- `application.py` : exemples d'applications pratiques.

## Exemple de transformée
```python
fft = np.fft.fft2(image)
fft_shift = np.fft.fftshift(fft)
amp = np.log1p(np.abs(fft_shift))
```

## Applications

- Compression d'images
- Suppression de bruit fréquentiel
- Détection de motifs périodiques

Pour plus d'informations : [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html).

