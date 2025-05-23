# TP5 : Amélioration par histogramme

Ce dossier présente plusieurs transformations d'intensité : correction gamma, étirement de contraste et traitements basés sur l'histogramme.

## Principes

- **LUT (Look-Up Table)** : transformation pixel par pixel.
- **Égalisation d'histogramme** : redistribution des niveaux de gris pour augmenter le contraste.
- **Matching d'histogramme** : adaptation à l'histogramme d'une image de référence.
Ces techniques modifient la distribution des niveaux de gris afin d'etendre la dynamique de l'image. L'egalisation s'appuie sur la theorie des probabilites pour uniformiser l'histogramme, tandis que le matching transfere la distribution d'une image de reference vers celle traitee.


## Scripts

- `gamma_correction.py` : application de différents gamma.
- `contrast_stretching.py` : étirement de contraste via la formule `1 / (1+(m/r)^E)`.
- `histogram_equalization.py` : égalisation manuelle et via scikit-image.
- `histogram_matching.py` : correspondance d'histogrammes.
- `lut_transformations.py` : visualisation des LUT.
- `combined_enhancement.py` : enchaînement de plusieurs méthodes.

## Exemple de correction gamma
```python
from skimage import exposure
corrigee = exposure.adjust_gamma(image, gamma=0.5)
```

## Utilisation

```bash
python run.py TP5_Enhancement/gamma_correction.py
```
Les figures sont sauvegardées dans `plots/TP5_Enhancement/`.

## Références

- Documentation scikit-image : [exposure](https://scikit-image.org/docs/stable/api/skimage.exposure.html)

