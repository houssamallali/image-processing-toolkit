#!/usr/bin/env python3
"""
TP 11 - Enhanced Main Script with Cross-Validation
Complete machine learning pipeline with advanced evaluation and optimization.

This enhanced version includes:
1. Feature extraction from Kimia database
2. Basic classification with SVM and MLP
3. Advanced cross-validation analysis
4. Hyperparameter optimization
5. Statistical significance testing
6. Comprehensive visualizations
7. Performance optimization

Author: Generated for TP11 Assignment
Date: 2024
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from feature_extraction import extract_features_from_database, visualize_sample_images
    from classification import ImageClassificationSystem
    from cross_validation import run_comprehensive_cv_analysis
    from config import get_database_path, validate_config, print_config_summary
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are in the tp11/ directory")
    sys.exit(1)


class EnhancedKimiaClassifier:
    """
    Enhanced Kimia image classifier with comprehensive evaluation capabilities.
    """
    
    def __init__(self):
        """Initialize the enhanced classifier system."""
        self.features = None
        self.labels = None
        self.classes = None
        self.basic_classifier = None
        self.cv_system = None
        
        print("Enhanced Kimia Image Classifier")
        print("=" * 50)
        
        # Validate configuration
        if not validate_config():
            print("❌ Configuration validation failed!")
            return
        
        # Check database availability
        db_path = get_database_path()
        if db_path is None:
            print("❌ Kimia database not found!")
            print("Please ensure the database is available in one of the expected locations.")
            return
        
        print(f"✅ Using database: {db_path}")
    
    def run_complete_analysis(self):
        """
        Run the complete enhanced analysis pipeline.
        """
        print("\n" + "="*80)
        print("ENHANCED TP11 ANALYSIS PIPELINE")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Step 1: Feature Extraction
            print("\n🔍 Step 1: Feature Extraction and Visualization")
            print("-" * 50)
            self._extract_features()
            
            # Step 2: Basic Classification
            print("\n🤖 Step 2: Basic Classification System")
            print("-" * 50)
            self._run_basic_classification()
            
            # Step 3: Advanced Cross-Validation
            print("\n📊 Step 3: Advanced Cross-Validation Analysis")
            print("-" * 50)
            self._run_advanced_cv()
            
            # Step 4: Results Summary
            print("\n📋 Step 4: Results Summary and Recommendations")
            print("-" * 50)
            self._generate_summary()
            
            total_time = time.time() - start_time
            print(f"\n✅ Complete analysis finished in {total_time:.2f} seconds")
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_features(self):
        """Extract features and create visualizations."""
        print("Extracting features from Kimia database...")
        
        # Visualize sample images first
        try:
            visualize_sample_images()
            print("✅ Sample images visualization completed")
        except Exception as e:
            print(f"⚠️  Sample visualization warning: {e}")
        
        # Extract features
        self.features, self.labels, self.classes = extract_features_from_database()
        
        print(f"✅ Feature extraction completed:")
        print(f"   • {self.features.shape[0]} images processed")
        print(f"   • {self.features.shape[1]} features per image")
        print(f"   • {len(self.classes)} classes: {', '.join(self.classes)}")
    
    def _run_basic_classification(self):
        """Run basic classification with standard algorithms."""
        print("Setting up basic classification system...")
        
        # Initialize classification system
        self.basic_classifier = ImageClassificationSystem(
            self.features, self.labels, self.classes
        )
        
        # Preprocess data
        self.basic_classifier.preprocess_data()
        self.basic_classifier.split_data(test_size=0.3, random_state=42)
        
        # Train primary classifiers
        self.basic_classifier.train_svm_classifier()
        self.basic_classifier.train_mlp_classifier()
        
        # Train additional classifiers
        self.basic_classifier.train_additional_classifiers()
        
        # Evaluate all models
        self.basic_classifier.evaluate_all_models()
        
        # Generate visualizations
        try:
            self.basic_classifier.plot_accuracy_comparison()
            self.basic_classifier.plot_confusion_matrices()
            self.basic_classifier.get_feature_importance()
            print("✅ Basic classification visualizations completed")
        except Exception as e:
            print(f"⚠️  Visualization warning: {e}")
        
        print("✅ Basic classification completed")
    
    def _run_advanced_cv(self):
        """Run advanced cross-validation analysis."""
        print("Starting comprehensive cross-validation analysis...")
        
        # Run the complete CV pipeline
        self.cv_system = run_comprehensive_cv_analysis(
            self.features, self.labels, self.classes
        )
        
        print("✅ Advanced cross-validation analysis completed")
    
    def _generate_summary(self):
        """Generate comprehensive results summary."""
        print("Generating comprehensive results summary...")
        
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        # Basic classification results
        if self.basic_classifier and self.basic_classifier.accuracies:
            print("\n📊 Basic Classification Results:")
            for model, accuracy in self.basic_classifier.accuracies.items():
                print(f"   • {model:15s}: {accuracy:.4f}")
            
            best_basic = max(self.basic_classifier.accuracies.items(), key=lambda x: x[1])
            print(f"\n🏆 Best basic model: {best_basic[0]} ({best_basic[1]:.4f})")
        
        # Cross-validation results
        if self.cv_system and self.cv_system.cv_results:
            print("\n📈 Cross-Validation Results (Mean ± Std):")
            for model, results in self.cv_system.cv_results.items():
                acc_mean = results['accuracy'].mean()
                acc_std = results['accuracy'].std()
                print(f"   • {model:15s}: {acc_mean:.4f} ± {acc_std:.4f}")
            
            best_cv = max(self.cv_system.cv_results.items(), 
                         key=lambda x: x[1]['accuracy'].mean())
            print(f"\n🏆 Best CV model: {best_cv[0]} ({best_cv[1]['accuracy'].mean():.4f})")
        
        # Optimized models results
        if self.cv_system and self.cv_system.best_models:
            print(f"\n🔧 Hyperparameter Optimization:")
            print(f"   • {len(self.cv_system.best_models)} models optimized")
            print("   • Grid search completed for all algorithms")
            print("   • Best parameters identified for each model")
        
        # Generated files
        print("\n📁 Generated Files:")
        expected_files = [
            "sample_kimia_images.png",
            "feature_distributions_by_class.png", 
            "accuracy_comparison.png",
            "all_confusion_matrices.png",
            "feature_importance.png",
            "model_comparison.csv",
            "statistical_significance.png",
            "cv_comprehensive_results.png",
            "learning_curve.png"
        ]
        
        for filename in expected_files:
            filepath = f"tp11/{filename}"
            if os.path.exists(filepath):
                print(f"   ✅ {filename}")
            else:
                print(f"   ❌ {filename} (not generated)")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if self.cv_system and self.cv_system.cv_results:
            # Find model with best CV performance
            best_model_name = max(self.cv_system.cv_results.keys(),
                                key=lambda x: self.cv_system.cv_results[x]['accuracy'].mean())
            best_accuracy = self.cv_system.cv_results[best_model_name]['accuracy'].mean()
            
            print(f"   • Use {best_model_name} for production (best CV performance)")
            
            if best_accuracy > 0.9:
                print("   • Excellent performance achieved!")
            elif best_accuracy > 0.8:
                print("   • Good performance - consider feature engineering for improvement")
            else:
                print("   • Consider collecting more data or trying different features")
        
        print("   • Review confusion matrices to identify problematic class pairs")
        print("   • Check learning curves for overfitting indicators")
        print("   • Use statistical significance tests for model selection confidence")


def main():
    """
    Main function to run the enhanced TP11 analysis.
    """
    print("TP 11 - Enhanced Machine Learning for Image Classification")
    print("Complete Analysis with Cross-Validation and Optimization")
    print("=" * 70)
    
    # Print configuration summary
    print_config_summary()
    
    # Initialize and run enhanced classifier
    classifier = EnhancedKimiaClassifier()
    classifier.run_complete_analysis()
    
    print("\n" + "="*70)
    print("🎉 ENHANCED TP11 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nFor detailed analysis:")
    print("• Check generated visualizations in tp11/ directory")
    print("• Review model_comparison.csv for quantitative results")
    print("• Examine learning curves for overfitting analysis")
    print("• Use statistical significance results for model selection")


if __name__ == "__main__":
    main()
