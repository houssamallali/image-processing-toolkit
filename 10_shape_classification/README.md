# TP11 : Classification de formes

Ce TP illustre l'utilisation de techniques d'apprentissage automatique pour reconnaître des silhouettes issues de la base Kimia.

## Étapes principales

1. **Extraction de caractéristiques** : mesures géométriques (aire, périmètre, excentricité...) calculées sur les images binaires.
2. **Prétraitement** : normalisation des données et séparation entraînement/test.
3. **Algorithmes** : SVM à noyau RBF, perceptron multicouche, forêts aléatoires et K plus proches voisins.
4. **Évaluation** : matrice de confusion et taux de réussite pour comparer les méthodes.
La reconnaissance de formes repose sur l'extraction de descripteurs invariants caracterisant la geometrie des objets. Ces vecteurs servent ensuite a entrainer des classifieurs supervises fondees sur la theorie statistique de la decision.


Les scripts fournissent également des visualisations et un classement des caractéristiques les plus discriminantes.

Pour plus de détails théoriques sur chaque algorithme, se référer à la documentation scikit-learn.

