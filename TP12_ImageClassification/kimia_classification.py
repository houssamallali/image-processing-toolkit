import glob
import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Path to the Kimia image database
DATA_DIR = os.path.join('..', 'images', 'images_Kimia')

# List of classes (18 in the full database)
classes = ['bird', 'bone', 'brick', 'camel', 'car', 'children',
           'classic', 'elephant', 'face', 'fork', 'fountain',
           'glass', 'hammer', 'heart', 'key', 'misk', 'ray', 'turtle']

features = []
labels = []

for idx, cls in enumerate(classes):
    pattern = os.path.join(DATA_DIR, f'{cls}*')
    for filename in glob.glob(pattern):
        img = imread(filename)
        if img.ndim == 3:
            img = rgb2gray(img)
        img = img > 0
        props = regionprops(label(img))[0]
        feats = [
            props.area,
            props.perimeter,
            props.eccentricity,
            props.extent,
            props.solidity,
            props.major_axis_length,
            props.minor_axis_length,
            props.filled_area,
            props.euler_number,
        ]
        features.append(feats)
        labels.append(idx)

X = np.array(features)
y = np.array(labels)

# One-hot target matrix if needed
nb_classes = len(classes)
Y = np.zeros((nb_classes, len(y)))
for i, label_idx in enumerate(y):
    Y[label_idx, i] = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=0)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print('Confusion matrix:')
print(confusion_matrix(y_test, pred))
print('\nClassification report:')
print(classification_report(y_test, pred, target_names=classes))
print(f'Accuracy: {accuracy_score(y_test, pred):.2f}')
