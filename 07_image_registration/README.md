# TP7 : Enregistrement d'images

Ce TP met en œuvre l'algorithme ICP (Iterative Closest Point) ainsi qu'une sélection manuelle de points pour aligner deux images.

## Principe

On cherche la rotation `R` et la translation `t` qui minimisent la distance quadratique entre les points correspondants :
\[ C(R,t) = \sum_i \| q_i - (R p_i + t) \|^2 \]
L'alignement d'images est un probleme d'optimisation non lineaire ou l'on cherche la transformation geometrique qui superpose au mieux deux ensembles de points ou d'intensites. L'algorithme ICP procède par estimations successives des correspondances et de la transformation jusqu'a convergence.


## Scripts

- `image_registration.py` : implémentation de l'ICP et des transformations rigides.
- `manual_registration_test.py` : sélection manuelle de points et visualisation du résultat.

## Utilisation rapide

```bash
python manual_registration_test.py
```
Sélectionnez 4 points correspondants dans les deux images. Les résultats sont sauvegardés dans `../plots/TP7_Registration/`.

Dépendances : NumPy, Matplotlib, scikit-image, OpenCV.

