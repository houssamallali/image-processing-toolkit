#!/usr/bin/env python3
"""
TP 11 - Cross-Validation and Hyperparameter Optimization Module
Advanced model evaluation and optimization for the Kimia image classification system.

This module implements:
1. Stratified K-Fold Cross-Validation
2. Grid Search with Cross-Validation
3. Statistical significance testing
4. Performance comparison with confidence intervals
5. Hyperparameter optimization for all algorithms

Author: Generated for TP11 Assignment
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score,
    cross_validate, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    CV_FOLDS, CV_SCORING, GRID_SEARCH_CV, GRID_SEARCH_SCORING,
    SVM_CONFIG, MLP_CONFIG, RF_CONFIG, KNN_CONFIG,
    SVM_ALTERNATIVES, MLP_ALTERNATIVES, RANDOM_STATE, N_JOBS,
    get_output_path, FIGURE_DPI
)


class CrossValidationSystem:
    """
    Comprehensive cross-validation and hyperparameter optimization system.
    """

    def __init__(self, X, y, class_names):
        """
        Initialize the cross-validation system.

        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            class_names (list): List of class names
        """
        self.X = X
        self.y = y
        self.class_names = class_names
        self.n_classes = len(class_names)

        # Preprocessing
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        # Cross-validation setup
        self.cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        # Results storage
        self.cv_results = {}
        self.best_models = {}
        self.grid_search_results = {}

        print(f"Cross-Validation System Initialized")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {self.n_classes} classes")
        print(f"CV Strategy: {CV_FOLDS}-Fold Stratified Cross-Validation")

    def perform_basic_cv(self):
        """
        Perform basic cross-validation for all algorithms with default parameters.
        """
        print("\n" + "="*60)
        print("BASIC CROSS-VALIDATION EVALUATION")
        print("="*60)

        # Define models with default configurations
        models = {
            'SVM': SVC(**SVM_CONFIG),
            'MLP': MLPClassifier(**MLP_CONFIG),
            'Random Forest': RandomForestClassifier(**RF_CONFIG),
            'KNN': KNeighborsClassifier(**KNN_CONFIG)
        }

        # Perform cross-validation for each model
        for name, model in models.items():
            print(f"\nEvaluating {name}...")

            # Multiple scoring metrics
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            cv_scores = cross_validate(
                model, self.X_scaled, self.y,
                cv=self.cv, scoring=scoring, n_jobs=N_JOBS
            )

            # Store results
            self.cv_results[name] = {
                'accuracy': cv_scores['test_accuracy'],
                'precision': cv_scores['test_precision_macro'],
                'recall': cv_scores['test_recall_macro'],
                'f1': cv_scores['test_f1_macro'],
                'fit_time': cv_scores['fit_time'],
                'score_time': cv_scores['score_time']
            }

            # Print summary statistics
            acc_mean = cv_scores['test_accuracy'].mean()
            acc_std = cv_scores['test_accuracy'].std()

            print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
            print(f"  Precision: {cv_scores['test_precision_macro'].mean():.4f} ± {cv_scores['test_precision_macro'].std():.4f}")
            print(f"  Recall: {cv_scores['test_recall_macro'].mean():.4f} ± {cv_scores['test_recall_macro'].std():.4f}")
            print(f"  F1-Score: {cv_scores['test_f1_macro'].mean():.4f} ± {cv_scores['test_f1_macro'].std():.4f}")
            print(f"  Training Time: {cv_scores['fit_time'].mean():.3f}s ± {cv_scores['fit_time'].std():.3f}s")

        return self.cv_results

    def perform_grid_search_optimization(self):
        """
        Perform grid search hyperparameter optimization for all algorithms.
        """
        print("\n" + "="*60)
        print("GRID SEARCH HYPERPARAMETER OPTIMIZATION")
        print("="*60)

        # Define parameter grids
        param_grids = {
            'SVM': [
                {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]},
                {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
                {'kernel': ['poly'], 'degree': [2, 3, 4], 'C': [0.1, 1, 10]}
            ],
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (100, 50, 25)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }

        # Define base models
        base_models = {
            'SVM': SVC(random_state=RANDOM_STATE, probability=True),
            'MLP': MLPClassifier(random_state=RANDOM_STATE, early_stopping=True),
            'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
            'KNN': KNeighborsClassifier()
        }

        # Perform grid search for each model
        for name in param_grids.keys():
            print(f"\nOptimizing {name}...")

            grid_search = GridSearchCV(
                base_models[name],
                param_grids[name],
                cv=GRID_SEARCH_CV,
                scoring=GRID_SEARCH_SCORING,
                n_jobs=N_JOBS,
                verbose=1
            )

            # Fit grid search
            grid_search.fit(self.X_scaled, self.y)

            # Store results
            self.grid_search_results[name] = grid_search
            self.best_models[name] = grid_search.best_estimator_

            print(f"  Best Score: {grid_search.best_score_:.4f}")
            print(f"  Best Parameters: {grid_search.best_params_}")

        return self.grid_search_results

    def compare_optimized_models(self):
        """
        Compare optimized models using cross-validation.
        """
        print("\n" + "="*60)
        print("OPTIMIZED MODELS COMPARISON")
        print("="*60)

        if not self.best_models:
            print("No optimized models found. Run grid search first.")
            return None

        optimized_results = {}

        for name, model in self.best_models.items():
            print(f"\nEvaluating optimized {name}...")

            # Cross-validation with multiple metrics
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            cv_scores = cross_validate(
                model, self.X_scaled, self.y,
                cv=self.cv, scoring=scoring, n_jobs=N_JOBS
            )

            optimized_results[name] = cv_scores

            # Print results
            acc_mean = cv_scores['test_accuracy'].mean()
            acc_std = cv_scores['test_accuracy'].std()
            print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")

        return optimized_results

    def statistical_significance_test(self):
        """
        Perform statistical significance tests between models.
        """
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*60)

        if not self.cv_results:
            print("No CV results found. Run basic CV first.")
            return None

        # Extract accuracy scores for all models
        model_names = list(self.cv_results.keys())
        accuracy_scores = {name: self.cv_results[name]['accuracy'] for name in model_names}

        # Perform pairwise t-tests
        print("\nPairwise t-test results (p-values):")
        print("-" * 40)

        significance_matrix = np.zeros((len(model_names), len(model_names)))

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(
                        accuracy_scores[model1],
                        accuracy_scores[model2]
                    )
                    significance_matrix[i, j] = p_value

                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"{model1:15s} vs {model2:15s}: p = {p_value:.4f} {significance}")

        # Create significance heatmap
        plt.figure(figsize=(10, 8))
        mask = np.eye(len(model_names), dtype=bool)
        sns.heatmap(significance_matrix, mask=mask, annot=True, fmt='.4f',
                   xticklabels=model_names, yticklabels=model_names,
                   cmap='RdYlBu_r', center=0.05)
        plt.title('Statistical Significance Matrix (p-values)\nLower values indicate more significant differences')
        plt.tight_layout()
        plt.savefig(get_output_path('statistical_significance'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return significance_matrix

    def plot_cv_results(self):
        """
        Create comprehensive visualizations of cross-validation results.
        """
        print("\n" + "="*60)
        print("GENERATING CROSS-VALIDATION VISUALIZATIONS")
        print("="*60)

        if not self.cv_results:
            print("No CV results found. Run basic CV first.")
            return

        # 1. Accuracy comparison with confidence intervals
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Accuracy comparison
        model_names = list(self.cv_results.keys())
        accuracies = [self.cv_results[name]['accuracy'] for name in model_names]

        axes[0, 0].boxplot(accuracies, labels=model_names)
        axes[0, 0].set_title('Cross-Validation Accuracy Distribution')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Performance metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_means = {metric: [self.cv_results[name][metric].mean() for name in model_names]
                      for metric in metrics}

        x = np.arange(len(model_names))
        width = 0.2

        for i, metric in enumerate(metrics):
            axes[0, 1].bar(x + i*width, metric_means[metric], width, label=metric.capitalize())

        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Performance Metrics Comparison')
        axes[0, 1].set_xticks(x + width * 1.5)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()

        # Training time comparison
        training_times = [self.cv_results[name]['fit_time'] for name in model_names]
        axes[1, 0].boxplot(training_times, labels=model_names)
        axes[1, 0].set_title('Training Time Distribution')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Accuracy vs Training Time scatter
        acc_means = [self.cv_results[name]['accuracy'].mean() for name in model_names]
        time_means = [self.cv_results[name]['fit_time'].mean() for name in model_names]

        axes[1, 1].scatter(time_means, acc_means, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (time_means[i], acc_means[i]),
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Mean Training Time (seconds)')
        axes[1, 1].set_ylabel('Mean Accuracy')
        axes[1, 1].set_title('Accuracy vs Training Time Trade-off')

        plt.tight_layout()
        plt.savefig(get_output_path('cv_comprehensive_results'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.show()

        # 2. Detailed performance report
        self._generate_performance_report()

    def _generate_performance_report(self):
        """
        Generate a detailed performance report.
        """
        print("\n" + "="*60)
        print("DETAILED PERFORMANCE REPORT")
        print("="*60)

        # Create performance summary table
        summary_data = []

        for name in self.cv_results.keys():
            results = self.cv_results[name]

            summary_data.append({
                'Model': name,
                'Accuracy_Mean': results['accuracy'].mean(),
                'Accuracy_Std': results['accuracy'].std(),
                'Precision_Mean': results['precision'].mean(),
                'Precision_Std': results['precision'].std(),
                'Recall_Mean': results['recall'].mean(),
                'Recall_Std': results['recall'].std(),
                'F1_Mean': results['f1'].mean(),
                'F1_Std': results['f1'].std(),
                'Training_Time_Mean': results['fit_time'].mean(),
                'Training_Time_Std': results['fit_time'].std()
            })

        # Convert to DataFrame and save
        df = pd.DataFrame(summary_data)

        # Display formatted table
        print("\nPerformance Summary:")
        print("-" * 100)
        for _, row in df.iterrows():
            print(f"{row['Model']:15s} | "
                  f"Acc: {row['Accuracy_Mean']:.4f}±{row['Accuracy_Std']:.4f} | "
                  f"Prec: {row['Precision_Mean']:.4f}±{row['Precision_Std']:.4f} | "
                  f"Rec: {row['Recall_Mean']:.4f}±{row['Recall_Std']:.4f} | "
                  f"F1: {row['F1_Mean']:.4f}±{row['F1_Std']:.4f} | "
                  f"Time: {row['Training_Time_Mean']:.3f}±{row['Training_Time_Std']:.3f}s")

        # Save to CSV
        df.to_csv(get_output_path('model_comparison'), index=False)
        print(f"\nDetailed results saved to: {get_output_path('model_comparison')}")

        # Find best model
        best_model = df.loc[df['Accuracy_Mean'].idxmax(), 'Model']
        best_accuracy = df.loc[df['Accuracy_Mean'].idxmax(), 'Accuracy_Mean']

        print(f"\nBest performing model: {best_model} (Accuracy: {best_accuracy:.4f})")

        return df

    def learning_curve_analysis(self, model_name='SVM'):
        """
        Analyze learning curves for a specific model.

        Args:
            model_name (str): Name of the model to analyze
        """
        from sklearn.model_selection import learning_curve

        print(f"\n" + "="*60)
        print(f"LEARNING CURVE ANALYSIS - {model_name}")
        print("="*60)

        # Get the model
        if model_name in self.best_models:
            model = self.best_models[model_name]
        else:
            # Use default model
            models = {
                'SVM': SVC(**SVM_CONFIG),
                'MLP': MLPClassifier(**MLP_CONFIG),
                'Random Forest': RandomForestClassifier(**RF_CONFIG),
                'KNN': KNeighborsClassifier(**KNN_CONFIG)
            }
            model = models.get(model_name, models['SVM'])

        # Generate learning curve
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, self.X_scaled, self.y,
            train_sizes=train_sizes, cv=self.cv,
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        )

        # Calculate means and standard deviations
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(get_output_path('learning_curve'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.show()

        # Analysis
        final_gap = train_mean[-1] - val_mean[-1]
        print(f"Final training-validation gap: {final_gap:.4f}")

        if final_gap > 0.1:
            print("⚠️  Large gap suggests overfitting. Consider:")
            print("   - Reducing model complexity")
            print("   - Adding regularization")
            print("   - Collecting more data")
        elif final_gap < 0.02:
            print("✅ Good generalization performance")
        else:
            print("📊 Moderate generalization gap")

        return train_sizes_abs, train_scores, val_scores


def run_comprehensive_cv_analysis(X, y, class_names):
    """
    Run the complete cross-validation analysis pipeline.

    Args:
        X (np.array): Feature matrix
        y (np.array): Target labels
        class_names (list): List of class names

    Returns:
        CrossValidationSystem: Configured CV system with results
    """
    print("="*80)
    print("COMPREHENSIVE CROSS-VALIDATION ANALYSIS")
    print("TP11 - Machine Learning for Image Classification")
    print("="*80)

    # Initialize CV system
    cv_system = CrossValidationSystem(X, y, class_names)

    # Step 1: Basic cross-validation
    print("\n🔄 Step 1: Basic Cross-Validation...")
    cv_system.perform_basic_cv()

    # Step 2: Hyperparameter optimization
    print("\n🔧 Step 2: Hyperparameter Optimization...")
    cv_system.perform_grid_search_optimization()

    # Step 3: Compare optimized models
    print("\n📊 Step 3: Optimized Models Comparison...")
    cv_system.compare_optimized_models()

    # Step 4: Statistical significance testing
    print("\n📈 Step 4: Statistical Significance Testing...")
    cv_system.statistical_significance_test()

    # Step 5: Generate visualizations
    print("\n📋 Step 5: Generating Comprehensive Visualizations...")
    cv_system.plot_cv_results()

    # Step 6: Learning curve analysis for best model
    print("\n📚 Step 6: Learning Curve Analysis...")
    cv_system.learning_curve_analysis('Random Forest')  # Usually best performer

    print("\n" + "="*80)
    print("✅ COMPREHENSIVE CROSS-VALIDATION ANALYSIS COMPLETED")
    print("="*80)
    print("\nGenerated files:")
    print("- model_comparison.csv (detailed performance metrics)")
    print("- statistical_significance.png (significance testing)")
    print("- cv_comprehensive_results.png (CV visualizations)")
    print("- learning_curve.png (learning curve analysis)")

    return cv_system


def main():
    """
    Main function to demonstrate cross-validation capabilities.
    """
    print("TP 11 - Cross-Validation and Hyperparameter Optimization Demo")
    print("=" * 70)

    # This would typically load features from feature extraction
    try:
        from feature_extraction import extract_features_from_database
        print("Loading features from Kimia database...")
        features, labels, classes = extract_features_from_database()

        # Run comprehensive analysis
        cv_system = run_comprehensive_cv_analysis(features, labels, classes)

        print("\n🎯 Key Findings:")
        if cv_system.cv_results:
            best_model = max(cv_system.cv_results.keys(),
                           key=lambda x: cv_system.cv_results[x]['accuracy'].mean())
            best_acc = cv_system.cv_results[best_model]['accuracy'].mean()
            print(f"   • Best model: {best_model} ({best_acc:.4f} accuracy)")

        if cv_system.best_models:
            print(f"   • {len(cv_system.best_models)} models optimized via grid search")

        print("   • Statistical significance testing completed")
        print("   • Learning curves analyzed for overfitting detection")

    except ImportError:
        print("Feature extraction module not available.")
        print("Creating synthetic data for demonstration...")

        # Create synthetic data for demo
        np.random.seed(42)
        n_samples, n_features, n_classes = 200, 9, 5
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        class_names = [f'Class_{i}' for i in range(n_classes)]

        # Run analysis on synthetic data
        cv_system = run_comprehensive_cv_analysis(X, y, class_names)

        print("\n📝 Note: This demo used synthetic data.")
        print("   For real results, ensure the Kimia database is available.")


if __name__ == "__main__":
    main()
