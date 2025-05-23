# TP12: Kimia Image Classification

This module demonstrates feature extraction and classification on the Kimia 216 database. The images are binary silhouettes divided into 18 classes with 12 samples each.  Features are extracted for every image using `skimage.measure.regionprops`. The resulting feature matrix is used to train a neural network classifier.

The main steps are:

1. Load all images from the dataset.
2. Compute a set of geometric features for each image.
3. Build a feature matrix with shape `(n_samples, n_features)` and construct one‑hot class labels.
4. Split the data into training and test subsets (75/25).
5. Train an `MLPClassifier` and evaluate the accuracy using a confusion matrix.

Run `kimia_classification.py` directly or via `run.py` to reproduce the experiment.
