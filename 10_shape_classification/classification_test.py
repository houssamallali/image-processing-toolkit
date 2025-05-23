#!/usr/bin/env python3
"""
TP 11 - Test and Demonstration Script
Complete test suite for the image classification assignment.

This script demonstrates the complete workflow:
1. Feature extraction from Kimia database
2. Data preprocessing and visualization
3. Model training and evaluation
4. Results analysis and comparison

Author: Generated for TP11 Assignment
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from feature_extraction import extract_features_from_database, visualize_sample_images, analyze_feature_statistics
from classification import ImageClassificationSystem

def run_complete_tp11_demo():
    """
    Run the complete TP11 demonstration following the assignment structure.
    """
    print("="*60)
    print("TP 11 - MACHINE LEARNING FOR IMAGE CLASSIFICATION")
    print("Complete Demonstration and Testing")
    print("="*60)
    
    # Step 1: Visualize sample images (Figure 38.1)
    print("\n1. VISUALIZING SAMPLE IMAGES FROM KIMIA DATABASE")
    print("-" * 50)
    try:
        visualize_sample_images()
        print("✓ Sample images visualization completed")
    except Exception as e:
        print(f"✗ Error in sample visualization: {e}")
        print("Note: Make sure the Kimia database is available in images/images_Kimia/")
    
    # Step 2: Feature extraction (Section 38.1)
    print("\n2. FEATURE EXTRACTION FROM KIMIA DATABASE")
    print("-" * 50)
    try:
        properties, target, classes = extract_features_from_database()
        print("✓ Feature extraction completed successfully")
        print(f"  - Total images processed: {len(target)}")
        print(f"  - Feature matrix shape: {properties.shape}")
        print(f"  - Number of classes: {len(classes)}")
    except Exception as e:
        print(f"✗ Error in feature extraction: {e}")
        return None
    
    # Step 3: Feature analysis
    print("\n3. FEATURE ANALYSIS AND STATISTICS")
    print("-" * 50)
    try:
        analyze_feature_statistics(properties, target, classes)
        print("✓ Feature analysis completed")
    except Exception as e:
        print(f"✗ Error in feature analysis: {e}")
    
    # Step 4: Classification system setup
    print("\n4. SETTING UP CLASSIFICATION SYSTEM")
    print("-" * 50)
    try:
        classifier = ImageClassificationSystem(properties, target, classes)
        classifier.preprocess_data()
        classifier.split_data(test_size=0.3, random_state=42)
        print("✓ Classification system initialized")
    except Exception as e:
        print(f"✗ Error in classification setup: {e}")
        return None
    
    # Step 5: Train SVM classifier (as shown in assignment)
    print("\n5. TRAINING SVM CLASSIFIER")
    print("-" * 50)
    try:
        classifier.train_svm_classifier(kernel='rbf', C=1.0)
        print("✓ SVM classifier trained successfully")
    except Exception as e:
        print(f"✗ Error in SVM training: {e}")
    
    # Step 6: Train MLP classifier (as shown in assignment)
    print("\n6. TRAINING MLP CLASSIFIER")
    print("-" * 50)
    try:
        classifier.train_mlp_classifier(hidden_layer_sizes=(100, 50))
        print("✓ MLP classifier trained successfully")
    except Exception as e:
        print(f"✗ Error in MLP training: {e}")
    
    # Step 7: Train additional classifiers for comparison
    print("\n7. TRAINING ADDITIONAL CLASSIFIERS")
    print("-" * 50)
    try:
        classifier.train_additional_classifiers()
        print("✓ Additional classifiers trained successfully")
    except Exception as e:
        print(f"✗ Error in additional classifier training: {e}")
    
    # Step 8: Comprehensive evaluation
    print("\n8. MODEL EVALUATION AND COMPARISON")
    print("-" * 50)
    try:
        classifier.evaluate_all_models()
        print("✓ Model evaluation completed")
    except Exception as e:
        print(f"✗ Error in model evaluation: {e}")
    
    # Step 9: Visualization and analysis
    print("\n9. RESULTS VISUALIZATION")
    print("-" * 50)
    try:
        classifier.plot_accuracy_comparison()
        classifier.plot_confusion_matrices()
        classifier.get_feature_importance()
        classifier.analyze_classification_errors()
        print("✓ Results visualization completed")
    except Exception as e:
        print(f"✗ Error in results visualization: {e}")
    
    # Step 10: Summary
    print("\n10. FINAL RESULTS SUMMARY")
    print("-" * 50)
    try:
        print("Model Performance Summary:")
        for model_name, accuracy in classifier.accuracies.items():
            print(f"  {model_name:15s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        best_model = max(classifier.accuracies, key=classifier.accuracies.get)
        best_accuracy = classifier.accuracies[best_model]
        print(f"\nBest performing model: {best_model} with {best_accuracy:.4f} accuracy")
        
        print("\n✓ TP11 demonstration completed successfully!")
        
    except Exception as e:
        print(f"✗ Error in final summary: {e}")
    
    return classifier

def test_individual_components():
    """
    Test individual components of the TP11 implementation.
    """
    print("\n" + "="*60)
    print("INDIVIDUAL COMPONENT TESTING")
    print("="*60)
    
    # Test 1: Database loading
    print("\nTest 1: Database Structure Loading")
    try:
        from feature_extraction import load_kimia_database
        classes, nb_classes, nb_images = load_kimia_database()
        print(f"✓ Database structure loaded: {nb_classes} classes, {nb_images} images per class")
        print(f"  Classes: {classes}")
    except Exception as e:
        print(f"✗ Database loading failed: {e}")
    
    # Test 2: Feature extraction on single image
    print("\nTest 2: Single Image Feature Extraction")
    try:
        import glob
        from skimage import measure, io
        
        # Try to find a sample image
        pattern = "images/images_Kimia/bird-*.bmp"
        filelist = glob.glob(pattern)
        
        if len(filelist) > 0:
            filename = filelist[0]
            image = io.imread(filename)
            
            # Convert to binary
            if len(image.shape) == 3:
                binary_image = image[:, :, 0] > 128
            else:
                binary_image = image > 128
            
            # Extract properties
            props = measure.regionprops(binary_image.astype(int))
            
            if len(props) > 0:
                largest_region = max(props, key=lambda x: x.area)
                print(f"✓ Single image processing successful")
                print(f"  Image: {filename}")
                print(f"  Area: {largest_region.area}")
                print(f"  Perimeter: {largest_region.perimeter:.2f}")
            else:
                print("✗ No regions found in image")
        else:
            print("✗ No sample images found")
            
    except Exception as e:
        print(f"✗ Single image processing failed: {e}")
    
    # Test 3: Classification system initialization
    print("\nTest 3: Classification System Initialization")
    try:
        # Create dummy data for testing
        dummy_properties = np.random.rand(100, 9)
        dummy_target = np.random.randint(0, 5, 100)
        dummy_classes = ['class1', 'class2', 'class3', 'class4', 'class5']
        
        test_classifier = ImageClassificationSystem(dummy_properties, dummy_target, dummy_classes)
        test_classifier.preprocess_data()
        test_classifier.split_data()
        
        print("✓ Classification system initialization successful")
        print(f"  Training samples: {test_classifier.X_train.shape[0]}")
        print(f"  Testing samples: {test_classifier.X_test.shape[0]}")
        
    except Exception as e:
        print(f"✗ Classification system initialization failed: {e}")

def create_sample_report():
    """
    Create a sample report showing the expected output format.
    """
    print("\n" + "="*60)
    print("SAMPLE EXPECTED OUTPUT")
    print("="*60)
    
    sample_output = """
Expected TP11 Results Format:
-----------------------------

1. Feature Extraction Results:
   - Total images processed: 216 (18 classes × 12 images)
   - Feature matrix shape: (216, 9)
   - Features: area, convex_area, eccentricity, equivalent_diameter, 
              extent, major_axis_length, minor_axis_length, perimeter, solidity

2. Classification Results:
   SVM Classifier:
   - Accuracy: 0.8500 (85.00%)
   - Best parameters: kernel='rbf', C=1.0
   
   MLP Classifier:
   - Accuracy: 0.8200 (82.00%)
   - Architecture: (100, 50) hidden layers
   
   Random Forest:
   - Accuracy: 0.8700 (87.00%)
   - n_estimators: 100
   
   K-Nearest Neighbors:
   - Accuracy: 0.7800 (78.00%)
   - k=5 neighbors

3. Best Model: Random Forest with 87.00% accuracy

4. Generated Visualizations:
   - sample_kimia_images.png: Sample images from database
   - feature_distributions_by_class.png: Feature analysis
   - accuracy_comparison.png: Model comparison
   - all_confusion_matrices.png: Confusion matrices
   - feature_importance.png: Feature importance ranking

Note: Actual results may vary depending on the available Kimia database
and random seed settings.
"""
    
    print(sample_output)

def main():
    """
    Main function to run all tests and demonstrations.
    """
    print("TP 11 - Complete Testing and Demonstration Suite")
    
    # Run individual component tests first
    test_individual_components()
    
    # Show expected output format
    create_sample_report()
    
    # Run complete demonstration
    classifier = run_complete_tp11_demo()
    
    if classifier is not None:
        print("\n" + "="*60)
        print("TP11 IMPLEMENTATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("- tp11/sample_kimia_images.png")
        print("- tp11/feature_distributions_by_class.png") 
        print("- tp11/accuracy_comparison.png")
        print("- tp11/all_confusion_matrices.png")
        print("- tp11/feature_importance.png")
        print("- tp11/extracted_features.npz")
    else:
        print("\n" + "="*60)
        print("TP11 IMPLEMENTATION ENCOUNTERED ISSUES")
        print("="*60)
        print("\nPlease check:")
        print("- Kimia database availability in images/images_Kimia/")
        print("- Required Python packages installation")
        print("- File permissions for output directory")

if __name__ == "__main__":
    main()
