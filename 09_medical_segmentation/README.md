# TP9 : Segmentation de follicules

Ce TP traite la segmentation d'images histologiques d'ovaire afin d'extraire et de mesurer différentes zones du follicule :
1. **Antrum** : cavité centrale.
2. **Thèque** : zone annulaire autour de l'antrum.
3. **Vascularisation** : vaisseaux sanguins dans la thèque.
4. **Cellules de granulosa** : entre l'antrum et la vascularisation.

## Méthode

Combinaison de seuillage et de morphologie mathématique :
- Extraction de l'antrum par seuillage sur le canal bleu.
- Dilatation binaire pour isoler la thèque.
- Recherche de zones sombres pour la vascularisation.
- Calcul des surfaces et proportions des différentes régions.

## Fichiers

- `follicle_segmentation.py` : implémentation principale.
- `main.py` : script d'exécution.

## Exécution

```bash
python main.py
```

Les résultats (images segmentées et mesures) sont stockés dans `plots/TP9_Follicle_Segmentation/`.

