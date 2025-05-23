# Projet de Traitement d'Images

Ce dépôt regroupe une série de scripts Python couvrant de nombreuses thématiques du traitement d'images. Chaque dossier `TPx` correspond à un thème de travaux pratiques.
## Bases theoriques

Le traitement d'images s'appuie sur la representation matricielle des pixels. Les filtrages spatiaux manipulent cette matrice via la convolution avec un noyau. L'analyse de Fourier permet de travailler dans le domaine frequentiel pour isoler les basses et hautes frequences. La segmentation repose sur des criteres d'intensite ou de texture pour regrouper les pixels en regions homogenes. Enfin la restauration modele la degradation comme une convolution avec une PSF suivie d'un bruit et applique des filtres inverses ou regularises pour recuperer l'image d'origine.


## Organisation du projet

- **TP1_Basics** : opérations élémentaires
  - `firsttest.py` : chargement, affichage et sauvegarde avec manipulation des canaux RGB
- **TP2_Filtering** : filtrage spatial
  - `lowpass.py`, `highpass.py`, `convolution.py`, `enhancement.py`, `aliasing_effect.py`
- **TP3_FourierAnalysis** : analyse par transformée de Fourier
  - `2Dfourier.py`, `inversefourier.py`, `LP_and_HP_filtering.py`, `application.py`
- **TP4_Segmentation** : techniques de segmentation d'image
  - `thresholding.py`, `kmeans_segmentation.py`, `otsu_segmentation.py`, `advanced_segmentation.py`
- **TP5_Enhancement** : amélioration de l'image
  - `gamma_correction.py`, `contrast_stretching.py`, `histogram_enhancement.py`, `lut_transformations.py`, `combined_enhancement.py`, `phobos_synthetic.py`
- **TP6_Restoration** : restauration et déconvolution
  - `image_restoration.py`, `create_motion_psf.py`, `astronomy_restoration.py`, `create_sample_astronomy.py`, `iterative_restoration.py`
- **TP7_Registration** : algorithmes d'alignement d'images
- **TP8_Compression** : méthodes de compression
- **TP9_Follicle_Segmentation** : segmentation de follicules ovariens
- **images** : toutes les images utilisées par les scripts
- **plots** : visualisations générées (par dossier TP)
- **docs** : documentation complémentaire

## Exécution des scripts

Le script `run.py` permet de lancer facilement n'importe quel script :

```bash
# Lister les scripts disponibles
python run.py --list

# Lancer un script précis
python run.py TP2_Filtering/lowpass.py

# Sauvegarder automatiquement les figures
python run.py TP3_FourierAnalysis/inversefourier.py --save-plots

# Exécuter tous les scripts d'un TP
python run.py --all-tp 9
```

Les figures produites sont enregistrées dans le dossier `plots`.

## Utilisation directe

Chaque fichier Python peut être exécuté individuellement. Les chemins d'accès aux images sont relatifs au répertoire `images` :

```python
from skimage.io import imread
image = imread('../images/exemple.jpg')
```

## Dépendances

Les bibliothèques nécessaires sont listées dans `requirements.txt` :

```bash
pip install -r requirements.txt
```

Principales dépendances : NumPy, Matplotlib, scikit-image, SciPy, scikit-learn et OpenCV (via `opencv-python`).

