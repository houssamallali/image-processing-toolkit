# TP4 : Segmentation d'images

La segmentation découpe une image en régions cohérentes pour en faciliter l'analyse.

## Concepts clés

- **Segmentation** : division de l'image en zones significatives.
- **Seuillage** : séparation par valeur d'intensité.
- **Clustering** : regroupement de pixels similaires (ex. K-means).
- **Croissance de région** : expansion à partir de points de départ.
- **Watershed** : approche basée sur la topographie.

## Fichiers principaux

- `thresholding.py` : seuillage manuel.
- `kmeans_segmentation.py` : segmentation automatique par K-means.
- `otsu_segmentation.py` : choix de seuil automatique (méthode d'Otsu).
- `advanced_segmentation.py` : méthodes plus élaborées.

## Exemple de seuillage
```python
seuil = 100
binaire = image > seuil
```

## Applications courantes

- Analyse médicale (organes, tumeurs)
- Comptage d'objets en microscopie
- Détection de défauts industriels

Pour un aperçu interactif : [Segmentation scikit-image](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html).

