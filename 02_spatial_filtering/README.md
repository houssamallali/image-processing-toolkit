# TP2 : Filtrage spatial

Ce module présente différents filtres appliqués directement sur les pixels pour améliorer une image, réduire le bruit ou extraire des détails.

## Théorie

La convolution consiste à combiner l'image avec un noyau pour produire un résultat filtré :

```
g(x,y) = \sum_i \sum_j f(i,j)\,h(x-i, y-j)
```

- `f` : image d'entrée
- `h` : noyau
- `g` : image filtrée

Types de filtres :
- **Passe-bas** : atténue les hautes fréquences (bruit), ex. moyenne, gaussien, médian.
- **Passe-haut** : met en valeur les contours, ex. Laplacien, Sobel.
- **Passe-bande** : conserve une bande de fréquences spécifique.

## Fichiers

- `lowpass.py` : exemples de filtres passe-bas.
- `highpass.py` : filtrage passe-haut pour accentuer les bords.
- `convolution.py` : convolution personnalisée avec différents noyaux.
- `enhancement.py` : techniques avancées d'amélioration.
- `aliasing_effect.py` : illustration de l'aliasing.

## Exemples

### Filtre moyen et gaussien
```python
from skimage.filters import gaussian
flou = gaussian(image, sigma=1.5)
```

### Filtre passe-haut (Laplacien)
```python
laplacien = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
haute = convolve2d(image, laplacien, mode='same', boundary='symm')
```

## Applications

- Réduction de bruit
- Détection de contours
- Netteté et suppression de flou

## Pour aller plus loin

- [Notes sur les filtres linéaires](https://www.cs.toronto.edu/~jepson/csc320/notes/linearFilters.pdf)
- [Tutoriel OpenCV sur le filtrage](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html)

