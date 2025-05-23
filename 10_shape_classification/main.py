#!/usr/bin/env python3
"""
TP 11 - Machine Learning for Image Classification
Objective: Classify images using machine learning techniques on the Kimia database

This implementation includes:
1. Feature extraction from binary images
2. Image classification using various ML algorithms
3. Performance evaluation and visualization

Author: Generated for TP11 Assignment
Date: 2024
"""

# Image manipulation and features construction
import glob
from skimage import measure, io
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing data and normalization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import QuantileTransformer

# Learning methods
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Plot confusion matrix
import seaborn as sns
import pandas as pd

class KimiaImageClassifier:
    """
    A complete image classification system for the Kimia database.
    
    This class handles feature extraction, model training, and evaluation
    for classifying binary images from the Kimia dataset.
    """
    
    def __init__(self, data_path="images/images_Kimia/"):
        """
        Initialize the classifier with the path to Kimia images.
        
        Args:
            data_path (str): Path to the directory containing Kimia images
        """
        self.data_path = data_path
        self.classes = ['bird', 'bone', 'brick', 'camel', 'car', 'children',
                       'classic', 'elephant', 'face', 'fork', 'fountain',
                       'glass', 'hammer', 'heart', 'key', 'misk', 'ray', 'turtle']
        self.nb_classes = len(self.classes)
        self.nb_images = 12  # Each class contains 12 images
        
        # Initialize arrays for features and targets
        self.properties = None
        self.target = None
        self.feature_names = ['area', 'convex_area', 'eccentricity', 
                             'equivalent_diameter', 'extent', 'major_axis_length',
                             'minor_axis_length', 'perimeter', 'solidity']
        
        # Models
        self.scaler = StandardScaler()
        self.svm_model = None
        self.mlp_model = None
        
    def extract_features(self):
        """
        Extract features from all images in the Kimia database.
        
        The 9 features extracted are:
        - area: Number of pixels in the region
        - convex_area: Number of pixels in the convex hull
        - eccentricity: Eccentricity of the ellipse
        - equivalent_diameter: Diameter of a circle with same area
        - extent: Ratio of pixels in region to pixels in bounding box
        - major_axis_length: Length of the major axis
        - minor_axis_length: Length of the minor axis
        - perimeter: Perimeter of the region
        - solidity: Ratio of pixels in region to pixels in convex hull
        """
        print("Starting feature extraction from Kimia database...")
        
        # Initialize feature matrix and target array
        total_images = self.nb_classes * self.nb_images
        nb_features = len(self.feature_names)
        self.properties = np.zeros((total_images, nb_features))
        self.target = np.zeros(total_images)
        
        index = 0
        
        # Process each class
        for class_idx, class_name in enumerate(self.classes):
            print(f"Processing class: {class_name}")
            
            # Get all files for this class
            pattern = f"{self.data_path}{class_name}-*.bmp"
            filelist = glob.glob(pattern)
            
            if len(filelist) == 0:
                print(f"Warning: No files found for pattern {pattern}")
                continue
                
            # Process each image in the class
            for filename in filelist[:self.nb_images]:  # Limit to nb_images per class
                try:
                    # Read the image
                    image = io.imread(filename)
                    
                    # Convert to binary if needed
                    if len(image.shape) > 2:
                        image = image[:,:,0] > 128
                    else:
                        image = image > 128
                    
                    # Extract region properties
                    props = measure.regionprops(image.astype(int))
                    
                    if len(props) > 0:
                        # Use the largest region
                        largest_region = max(props, key=lambda x: x.area)
                        
                        # Extract the 9 features
                        self.properties[index, 0] = largest_region.area
                        self.properties[index, 1] = largest_region.convex_area
                        self.properties[index, 2] = largest_region.eccentricity
                        self.properties[index, 3] = largest_region.equivalent_diameter
                        self.properties[index, 4] = largest_region.extent
                        self.properties[index, 5] = largest_region.major_axis_length
                        self.properties[index, 6] = largest_region.minor_axis_length
                        self.properties[index, 7] = largest_region.perimeter
                        self.properties[index, 8] = largest_region.solidity
                        
                        # Set target class
                        self.target[index] = class_idx
                        
                        index += 1
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
        
        # Trim arrays to actual number of processed images
        self.properties = self.properties[:index]
        self.target = self.target[:index]
        
        print(f"Feature extraction completed. Processed {index} images.")
        print(f"Feature matrix shape: {self.properties.shape}")
        
    def preprocess_data(self):
        """
        Preprocess the extracted features using standardization.
        """
        print("Preprocessing features...")
        self.properties = self.scaler.fit_transform(self.properties)
        print("Feature standardization completed.")
        
    def train_models(self, test_size=0.3, random_state=42):
        """
        Train SVM and MLP classifiers on the extracted features.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random state for reproducibility
        """
        print("Splitting data and training models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.properties, self.target, test_size=test_size, 
            random_state=random_state, stratify=self.target
        )
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Train SVM classifier
        print("Training SVM classifier...")
        self.svm_model = svm.SVC(kernel='rbf', random_state=random_state)
        self.svm_model.fit(X_train, y_train)
        
        # Train MLP classifier
        print("Training MLP classifier...")
        self.mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), 
                                      max_iter=1000, random_state=random_state)
        self.mlp_model.fit(X_train, y_train)
        
        print("Model training completed.")
        
    def evaluate_models(self):
        """
        Evaluate both SVM and MLP models and display results.
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Evaluate SVM
        svm_predictions = self.svm_model.predict(self.X_test)
        svm_accuracy = np.mean(svm_predictions == self.y_test)
        
        print(f"\nSVM Classifier Results:")
        print(f"Accuracy: {svm_accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, svm_predictions, 
                                  target_names=self.classes))
        
        # Evaluate MLP
        mlp_predictions = self.mlp_model.predict(self.X_test)
        mlp_accuracy = np.mean(mlp_predictions == self.y_test)
        
        print(f"\nMLP Classifier Results:")
        print(f"Accuracy: {mlp_accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, mlp_predictions, 
                                  target_names=self.classes))
        
        # Store predictions for visualization
        self.svm_predictions = svm_predictions
        self.mlp_predictions = mlp_predictions
        
        return svm_accuracy, mlp_accuracy
        
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for both SVM and MLP classifiers.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # SVM Confusion Matrix
        svm_cm = confusion_matrix(self.y_test, self.svm_predictions)
        sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes, ax=axes[0])
        axes[0].set_title('SVM Classifier - Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # MLP Confusion Matrix
        mlp_cm = confusion_matrix(self.y_test, self.mlp_predictions)
        sns.heatmap(mlp_cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=self.classes, yticklabels=self.classes, ax=axes[1])
        axes[1].set_title('MLP Classifier - Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('tp11/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_analysis(self):
        """
        Plot feature distributions and correlations.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature distributions
        for i, feature in enumerate(self.feature_names[:4]):
            axes[i//2, i%2].hist(self.properties[:, i], bins=30, alpha=0.7)
            axes[i//2, i%2].set_title(f'Distribution of {feature}')
            axes[i//2, i%2].set_xlabel(feature)
            axes[i//2, i%2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('tp11/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature correlation matrix
        plt.figure(figsize=(10, 8))
        correlation_matrix = np.corrcoef(self.properties.T)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=self.feature_names, yticklabels=self.feature_names)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('tp11/feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function to run the complete TP11 assignment.
    """
    print("TP 11 - Machine Learning for Image Classification")
    print("=" * 50)
    
    # Initialize the classifier
    classifier = KimiaImageClassifier()
    
    # Step 1: Extract features
    classifier.extract_features()
    
    # Step 2: Preprocess data
    classifier.preprocess_data()
    
    # Step 3: Train models
    classifier.train_models()
    
    # Step 4: Evaluate models
    svm_acc, mlp_acc = classifier.evaluate_models()
    
    # Step 5: Visualize results
    classifier.plot_confusion_matrices()
    classifier.plot_feature_analysis()
    
    print(f"\nFinal Results Summary:")
    print(f"SVM Accuracy: {svm_acc:.4f}")
    print(f"MLP Accuracy: {mlp_acc:.4f}")
    print(f"Best Model: {'SVM' if svm_acc > mlp_acc else 'MLP'}")

if __name__ == "__main__":
    main()
