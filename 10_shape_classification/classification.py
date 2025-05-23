#!/usr/bin/env python3
"""
TP 11 - Image Classification Module
Implementation of machine learning algorithms for image classification.

This module implements various classification algorithms as shown in the assignment:
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- Performance evaluation and comparison

Author: Generated for TP11 Assignment
"""

# Learning methods
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Additional classifiers for comparison
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class ImageClassificationSystem:
    """
    Complete image classification system implementing multiple ML algorithms.
    """
    
    def __init__(self, properties, target, classes):
        """
        Initialize the classification system.
        
        Args:
            properties (np.array): Feature matrix
            target (np.array): Target classes
            classes (list): Class names
        """
        self.properties = properties
        self.target = target
        self.classes = classes
        self.nb_classes = len(classes)
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.X_scaled = None
        
        # Train/test split
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Models
        self.models = {}
        self.predictions = {}
        self.accuracies = {}
        
    def preprocess_data(self):
        """
        Preprocess the feature data using standardization.
        """
        print("Preprocessing data...")
        self.X_scaled = self.scaler.fit_transform(self.properties)
        print("Data preprocessing completed.")
        
    def split_data(self, test_size=0.3, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
        """
        print(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.target, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=self.target
        )
        
        print(f"Training set size: {self.X_train.shape[0]} samples")
        print(f"Testing set size: {self.X_test.shape[0]} samples")
        
    def train_svm_classifier(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Train Support Vector Machine classifier.
        
        Args:
            kernel (str): SVM kernel type
            C (float): Regularization parameter
            gamma (str/float): Kernel coefficient
        """
        print("Training SVM classifier...")
        
        svm_model = svm.SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
        svm_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        svm_predictions = svm_model.predict(self.X_test)
        svm_accuracy = accuracy_score(self.y_test, svm_predictions)
        
        # Store results
        self.models['SVM'] = svm_model
        self.predictions['SVM'] = svm_predictions
        self.accuracies['SVM'] = svm_accuracy
        
        print(f"SVM training completed. Accuracy: {svm_accuracy:.4f}")
        
    def train_mlp_classifier(self, hidden_layer_sizes=(100, 50), max_iter=1000):
        """
        Train Multi-Layer Perceptron classifier.
        
        Args:
            hidden_layer_sizes (tuple): Size of hidden layers
            max_iter (int): Maximum number of iterations
        """
        print("Training MLP classifier...")
        
        mlp_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        mlp_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        mlp_predictions = mlp_model.predict(self.X_test)
        mlp_accuracy = accuracy_score(self.y_test, mlp_predictions)
        
        # Store results
        self.models['MLP'] = mlp_model
        self.predictions['MLP'] = mlp_predictions
        self.accuracies['MLP'] = mlp_accuracy
        
        print(f"MLP training completed. Accuracy: {mlp_accuracy:.4f}")
        
    def train_additional_classifiers(self):
        """
        Train additional classifiers for comparison.
        """
        print("Training additional classifiers...")
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        rf_predictions = rf_model.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_predictions)
        
        self.models['Random Forest'] = rf_model
        self.predictions['Random Forest'] = rf_predictions
        self.accuracies['Random Forest'] = rf_accuracy
        
        # K-Nearest Neighbors
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(self.X_train, self.y_train)
        knn_predictions = knn_model.predict(self.X_test)
        knn_accuracy = accuracy_score(self.y_test, knn_predictions)
        
        self.models['KNN'] = knn_model
        self.predictions['KNN'] = knn_predictions
        self.accuracies['KNN'] = knn_accuracy
        
        print(f"Random Forest accuracy: {rf_accuracy:.4f}")
        print(f"KNN accuracy: {knn_accuracy:.4f}")
        
    def evaluate_all_models(self):
        """
        Evaluate all trained models and display comprehensive results.
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        for model_name in self.models.keys():
            print(f"\n{model_name} Classifier Results:")
            print("-" * 40)
            print(f"Accuracy: {self.accuracies[model_name]:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(
                self.y_test, 
                self.predictions[model_name], 
                target_names=self.classes,
                zero_division=0
            ))
            
    def plot_accuracy_comparison(self):
        """
        Plot accuracy comparison between all models.
        """
        model_names = list(self.accuracies.keys())
        accuracies = list(self.accuracies.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'coral', 'gold'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add accuracy values on top of bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('tp11/accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for all models.
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        colors = ['Blues', 'Greens', 'Oranges', 'Purples']
        
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            if i < 4:  # Limit to 4 models for display
                cm = confusion_matrix(self.y_test, predictions)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap=colors[i], 
                           xticklabels=self.classes, yticklabels=self.classes, 
                           ax=axes[i])
                axes[i].set_title(f'{model_name} - Confusion Matrix\nAccuracy: {self.accuracies[model_name]:.3f}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(self.models), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('tp11/all_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_classification_errors(self):
        """
        Analyze classification errors for the best performing model.
        """
        # Find best model
        best_model_name = max(self.accuracies, key=self.accuracies.get)
        best_predictions = self.predictions[best_model_name]
        
        print(f"\nError Analysis for Best Model: {best_model_name}")
        print("=" * 50)
        
        # Find misclassified samples
        misclassified = self.y_test != best_predictions
        
        if np.sum(misclassified) > 0:
            print(f"Number of misclassified samples: {np.sum(misclassified)}")
            print(f"Error rate: {np.sum(misclassified)/len(self.y_test):.4f}")
            
            # Show confusion between classes
            print("\nMost common misclassifications:")
            for true_class in range(self.nb_classes):
                for pred_class in range(self.nb_classes):
                    if true_class != pred_class:
                        count = np.sum((self.y_test == true_class) & 
                                     (best_predictions == pred_class))
                        if count > 0:
                            print(f"  {self.classes[true_class]} → {self.classes[pred_class]}: {count} samples")
        else:
            print("Perfect classification! No errors found.")
            
    def get_feature_importance(self):
        """
        Get feature importance from Random Forest model if available.
        """
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_names = ['area', 'convex_area', 'eccentricity', 
                           'equivalent_diameter', 'extent', 'major_axis_length',
                           'minor_axis_length', 'perimeter', 'solidity']
            
            importance = rf_model.feature_importances_
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importance)[::-1]
            
            plt.bar(range(len(importance)), importance[indices])
            plt.title('Feature Importance (Random Forest)')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
            
            plt.tight_layout()
            plt.savefig('tp11/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\nFeature Importance Ranking:")
            for i, idx in enumerate(indices):
                print(f"{i+1:2d}. {feature_names[idx]:20s}: {importance[idx]:.4f}")

def main():
    """
    Main function to demonstrate the classification system.
    """
    print("TP 11 - Image Classification System")
    print("=" * 50)
    
    # This would typically load features from the feature extraction module
    # For demonstration, we'll create dummy data
    print("Note: This module requires features from feature_extraction.py")
    print("Run feature_extraction.py first to generate the required features.")
    
if __name__ == "__main__":
    main()
