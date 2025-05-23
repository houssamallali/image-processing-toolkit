# TP13 : Analyse multi-échelle

Ce module présente la décomposition pyramidale (gaussienne et laplacienne) ainsi que des approches morphologiques pour analyser les images à plusieurs résolutions.

L'analyse multi-echelle s'inspire de la theorie des ondelettes et de la morphologie mathematique pour representer les images a plusieurs niveaux de details. Les pyramides gaussiennes et laplaciennes offrent des approximations successives facilitant la detection de structures selon leur taille.

## Contenu

- `pyramidal_decomposition.py` : construction et reconstruction de pyramides.
- `morphological_multiscale.py` : décomposition par opérateurs morphologiques.
- `visualization.py` : outils de visualisation des niveaux et des erreurs.
- `config.py` : paramètres communs aux scripts.
- `main.py` : exécution complète de l'analyse.

### Exemple d'utilisation
```python
from tp13.pyramidal_decomposition import LaplacianPyramidDecomposition
image = load_cerveau_image()
G, L = LaplacianPyramidDecomposition(levels=4).decompose(image)
```

Les résultats (pyramides, reconstructions, mesures) sont enregistrés dans `tp13/outputs/`.

