# TP1 : Traitement d'image de base

Ce dossier introduit les opérations fondamentales du traitement d'images.

## Concepts clés

- **Représentation des images** : matrice de valeurs de pixels.
- **Canaux de couleur** : Rouge, Vert, Bleu pour les images couleur.
- **Entrée/Sortie** : chargement et sauvegarde de fichiers (JPG, PNG...).
- **Visualisation** : affichage des images et de leurs propriétés.

## Fichier principal

- `firsttest.py` : chargement d'une image, séparation des canaux RGB, affichage et sauvegarde avec différents niveaux de compression.

## Opérations essentielles

### Chargement
```python
from skimage.io import imread
image = imread('../images/retina.jpg')
```

### Propriétés
- Dimensions (hauteur, largeur, canaux)
- Type des données (uint8, float, etc.)

### Séparation des canaux
```python
rouge  = image[:, :, 0]
vert   = image[:, :, 1]
bleu   = image[:, :, 2]
```

### Compression
```python
from skimage.io import imsave
imsave('output.jpg', image, quality=50)  # qualité réduite
```

## Applications

- Visualisation médicale simple
- Analyse d'image élémentaire
- Conversion et optimisation de formats

## Contexte theorique

Une image numerique se definit comme une fonction discrete decrite par une matrice de valeurs. Les operations elementaires manipulent ces valeurs en modifiant l'espace de couleur ou leur distribution. Les conversions de format et la compression utilisent cette representation pour reduire la taille des fichiers tout en preservant l'information visuelle.

## Ressources utiles

- [Documentation scikit-image](https://scikit-image.org/docs/stable/)
- [Tutoriel Matplotlib](https://matplotlib.org/stable/tutorials/introductory/images.html)
- [Bibliothèque Pillow](https://pillow.readthedocs.io/en/stable/)

