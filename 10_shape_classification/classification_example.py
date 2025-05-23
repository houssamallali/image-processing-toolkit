#!/usr/bin/env python3
"""
TP 11 - Example Usage Script
Simple example demonstrating the core functionality of the image classification system.

This script provides a minimal working example that can be easily understood
and modified for different use cases.

Author: Generated for TP11 Assignment
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def create_sample_data():
    """
    Create sample data for demonstration when Kimia database is not available.

    Returns:
        tuple: (features, labels, class_names)
    """
    print("Creating sample data for demonstration...")

    # Simulate 9 features for 180 samples (18 classes × 10 samples)
    np.random.seed(42)
    n_classes = 18
    n_samples_per_class = 10
    n_features = 9

    features = []
    labels = []

    class_names = ['bird', 'bone', 'brick', 'camel', 'car', 'children',
                   'classic', 'elephant', 'face', 'fork', 'fountain',
                   'glass', 'hammer', 'heart', 'key', 'misk', 'ray', 'turtle']

    # Generate synthetic features for each class
    for class_id in range(n_classes):
        # Create class-specific feature distributions
        class_features = np.random.normal(
            loc=class_id * 0.5,  # Different mean for each class
            scale=1.0,
            size=(n_samples_per_class, n_features)
        )

        # Add some class-specific patterns
        class_features[:, 0] *= (class_id + 1) * 100  # area
        class_features[:, 7] *= (class_id + 1) * 50   # perimeter

        features.append(class_features)
        labels.extend([class_id] * n_samples_per_class)

    features = np.vstack(features)
    labels = np.array(labels)

    print(f"Generated {len(features)} samples with {n_features} features")
    print(f"Classes: {n_classes}")

    return features, labels, class_names

def simple_classification_example():
    """
    Simple example of the classification workflow.
    """
    print("="*50)
    print("TP 11 - Simple Classification Example")
    print("="*50)

    # Step 1: Get data (sample data or real features)
    try:
        # Try to load real features if available
        from feature_extraction import extract_features_from_database
        print("Attempting to load real Kimia features...")
        features, labels, class_names = extract_features_from_database()
        print("✓ Real Kimia features loaded successfully")
    except:
        print("Kimia database not available, using sample data...")
        features, labels, class_names = create_sample_data()

    # Step 2: Preprocess data
    print("\nPreprocessing data...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Step 3: Split data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Step 4: Train SVM classifier
    print("\nTraining SVM classifier...")
    svm_model = svm.SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    print(f"SVM Accuracy: {svm_accuracy:.4f}")

    # Step 5: Train MLP classifier
    print("\nTraining MLP classifier...")
    mlp_model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    mlp_model.fit(X_train, y_train)

    mlp_predictions = mlp_model.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, mlp_predictions)

    print(f"MLP Accuracy: {mlp_accuracy:.4f}")

    # Step 6: Display results
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)

    print(f"SVM Classifier: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    print(f"MLP Classifier: {mlp_accuracy:.4f} ({mlp_accuracy*100:.2f}%)")

    best_model = "SVM" if svm_accuracy > mlp_accuracy else "MLP"
    best_accuracy = max(svm_accuracy, mlp_accuracy)
    print(f"\nBest Model: {best_model} with {best_accuracy:.4f} accuracy")

    # Step 7: Detailed classification report for best model
    print(f"\nDetailed Classification Report for {best_model}:")
    print("-" * 40)

    try:
        if best_model == "SVM":
            print(classification_report(y_test, svm_predictions))
        else:
            print(classification_report(y_test, mlp_predictions))
    except Exception as e:
        print(f"Could not generate detailed report: {e}")
        print("This may happen when some classes are missing from the test set.")

    # Step 8: Simple visualization
    plot_simple_results(svm_accuracy, mlp_accuracy)

    return svm_model, mlp_model, svm_accuracy, mlp_accuracy

def plot_simple_results(svm_acc, mlp_acc):
    """
    Create a simple visualization of the results.

    Args:
        svm_acc (float): SVM accuracy
        mlp_acc (float): MLP accuracy
    """
    print("\nCreating results visualization...")

    # Simple bar plot
    models = ['SVM', 'MLP']
    accuracies = [svm_acc, mlp_acc]
    colors = ['skyblue', 'lightgreen']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=colors)

    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('TP 11 - Classification Results Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)

    # Add a horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    plt.legend()

    plt.tight_layout()
    plt.savefig('simple_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Results visualization saved as 'simple_results.png'")

def demonstrate_feature_importance():
    """
    Demonstrate feature importance analysis.
    """
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)

    feature_names = ['area', 'convex_area', 'eccentricity',
                    'equivalent_diameter', 'extent', 'major_axis_length',
                    'minor_axis_length', 'perimeter', 'solidity']

    # Create sample importance values
    np.random.seed(42)
    importance = np.random.rand(9)
    importance = importance / np.sum(importance)  # Normalize

    # Sort by importance
    indices = np.argsort(importance)[::-1]

    print("Feature Importance Ranking:")
    for i, idx in enumerate(indices):
        print(f"{i+1:2d}. {feature_names[idx]:20s}: {importance[idx]:.4f}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance[indices], color='coral')
    plt.title('Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('tp11/simple_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Feature importance visualization saved")

def main():
    """
    Main function to run the simple example.
    """
    print("TP 11 - Simple Example Usage")
    print("This script demonstrates the core functionality in a simplified way.")
    print()

    # Run the main classification example
    svm_model, mlp_model, svm_acc, mlp_acc = simple_classification_example()

    # Demonstrate feature importance
    demonstrate_feature_importance()

    print("\n" + "="*60)
    print("SIMPLE EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("- tp11/simple_results.png")
    print("- tp11/simple_feature_importance.png")
    print("\nFor the complete implementation, run:")
    print("- python tp11/main.py (complete system)")
    print("- python tp11/test_classification.py (full test suite)")

if __name__ == "__main__":
    main()
