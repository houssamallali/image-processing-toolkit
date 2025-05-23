# TP6 : Restauration d'images

On s'intéresse ici à la récupération d'images dégradées par un flou ou du bruit, notamment via la déconvolution.

## Modèle

Une image dégradée peut s'écrire :
```
g = h * f + n
```
avec `h` la fonction d'étalement (PSF) et `n` le bruit ajouté.
Ce modele provient de la theorie des systemes lineaires ou l'acquisition est assimilee a la convolution de l'image originale par une PSF. La restauration tente d'inverser ce processus en tenant compte du bruit pour retrouver l'image la plus proche possible de la verite terrain.


## Méthodes de restauration

- **Filtrage inverse** : division par `H` en fréquence.
- **Filtre régularisé** : ajoute une constante pour éviter les divisions par zéro.
- **Filtre de Wiener** : tient compte du bruit pour limiter l'amplification.
- **Méthodes itératives** : Richardson-Lucy, Van-Cittert, Landweber.

## Scripts

- `image_restoration.py` : démonstrations sur images synthétiques.
- `create_motion_psf.py` : génération de PSF de mouvement.
- `create_sample_astronomy.py` : images astronomiques simulées.
- `astronomy_restoration.py` : restauration de Jupiter et Saturne.
- `iterative_restoration.py` : comparaison des algorithmes itératifs.

## Exécution

```bash
python image_restoration.py
```
Les résultats sont enregistrés dans `plots/TP6_Restoration/`.

