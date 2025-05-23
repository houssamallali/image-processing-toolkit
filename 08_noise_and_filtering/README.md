# TP8 : Bruit aléatoire

Ce module génère différents types de bruit pour tester les algorithmes de filtrage.

Types de bruit :
1. **Uniforme** : valeurs réparties uniformément.
2. **Gaussien** : distribution normale autour d'une moyenne.
3. **Poivre et sel** : pixels noirs ou blancs aléatoires.
4. **Exponentiel** : modélise certains phénomènes de décroissance.

La generation de bruit suit des distributions de probabilite definies afin de simuler les perturbations observees lors de l'acquisition. Comprendre ces modeles permet d'evaluer la robustesse des filtres en presence de degradations controlees.

## Script

- `random_noise.py` : création et visualisation des bruits précédents.

## Lancer le script

```bash
python TP8_Compression/random_noise.py
```

Les images et histogrammes sont enregistrés dans `plots/TP8_Compression/`.

