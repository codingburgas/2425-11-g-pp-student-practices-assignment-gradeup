
"""
Demo script for the custom ML model implementation.
This script demonstrates all the features of the ML pipeline including:
1. Custom neural network training
2. Model evaluation with comprehensive metrics
3. Model persistence (save/load)
4. Integration with survey data
"""

import numpy as np
import os
import sys
from datetime import datetime


current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
sys.path.append(app_dir)

from ml.models import CustomNeuralNetwork
from ml.pipeline import MLTrainingPipeline
from ml.evaluator import ModelEvaluator
from ml.utils import create_sample_dataset, extract_features_from_survey_response


def demo_basic_neural_network():
    """Demonstrate basic neural network functionality."""
    print("=" * 60)
    print("DEMO 1: Basic Neural Network")
    print("=" * 60)
    
    
    print("Creating sample dataset...")
    X, y = create_sample_dataset(n_samples=1000, n_features=8, n_classes=3)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)}")
    
    
    print("\nInitializing neural network...")
    nn = CustomNeuralNetwork(
        input_size=8,
        hidden_sizes=[16, 8],
        output_size=3,
        activation='relu',
        output_activation='softmax',
        learning_rate=0.01,
        random_seed=42
    )
    
    
    y_onehot = np.zeros((len(y), 3))
    for i, label in enumerate(y):
        y_onehot[i, label] = 1
    
    print("Training neural network...")
    history = nn.train(X, y_onehot, epochs=50, batch_size=32, verbose=True)
    
    
    print("\nEvaluating model...")
    test_metrics = ModelEvaluator.evaluate_model(nn, X, y_onehot)
    
    print(f"Final Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision (macro): {test_metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {test_metrics['recall_macro']:.4f}")
    print(f"F1-score (macro): {test_metrics['f1_macro']:.4f}")
    print(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    
    model_path = "saved_models/demo_neural_network.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    nn.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    
    print("Loading saved model...")
    loaded_nn = CustomNeuralNetwork.load_model(model_path)
    loaded_predictions = loaded_nn.predict(X[:5])
    original_predictions = nn.predict(X[:5])
    
    print("Predictions match:", np.allclose(loaded_predictions, original_predictions))
    
    return nn, test_metrics


def demo_ml_pipeline():
    """Demonstrate the complete ML training pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 2: Complete ML Training Pipeline")
    print("=" * 60)
    
    
    print("Creating sample dataset...")
    X, y = create_sample_dataset(n_samples=1500, n_features=12, n_classes=4)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    
    print("\nInitializing ML training pipeline...")
    pipeline = MLTrainingPipeline(random_seed=42)
    
    
    print("Training model with pipeline...")
    results = pipeline.train_model(
        X, y,
        hidden_sizes=[32, 16, 8],
        activation='relu',
        learning_rate=0.005,
        epochs=80,
        batch_size=64,
        test_size=0.2,
        val_size=0.15,
        verbose=True
    )
    
    
    print("\n" + "-" * 40)
    print("DETAILED EVALUATION RESULTS")
    print("-" * 40)
    
    metrics = results['test_metrics']
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Precision (micro): {metrics['precision_micro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"Recall (micro): {metrics['recall_micro']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    print(f"F1-score (micro): {metrics['f1_micro']:.4f}")
    print(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")
    
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    
    print(f"\nPer-class Precision: {metrics['precision_per_class']}")
    print(f"Per-class Recall: {metrics['recall_per_class']}")
    print(f"Per-class F1-score: {metrics['f1_per_class']}")
    
    
    pipeline_path = "saved_models/demo_pipeline.pkl"
    pipeline.save_pipeline(pipeline_path)
    print(f"\nPipeline saved to: {pipeline_path}")
    
    
    print("\nTesting predictions on new data...")
    test_X = np.random.randn(3, 12)
    predictions = pipeline.predict(test_X)
    class_predictions = pipeline.predict_classes(test_X)
    
    print("Prediction probabilities:")
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: {pred}")
    print(f"Predicted classes: {class_predictions}")
    
    
    print("\nLoading saved pipeline...")
    loaded_pipeline = MLTrainingPipeline.load_pipeline(pipeline_path)
    loaded_predictions = loaded_pipeline.predict(test_X)
    
    print("Predictions match:", np.allclose(predictions, loaded_predictions))
    
    return pipeline, results


def demo_survey_integration():
    """Demonstrate integration with survey response data."""
    print("\n" + "=" * 60)
    print("DEMO 3: Survey Response Integration")
    print("=" * 60)
    
    
    sample_surveys = [
        {
            'math_interest': 8,
            'science_interest': 7,
            'art_interest': 3,
            'sports_interest': 5,
            'study_hours_per_day': 4,
            'preferred_subject': 'Mathematics',
            'career_goal': 'Engineer',
            'extracurricular': True,
            'leadership_experience': True,
            'team_preference': False,
            'languages_spoken': ['Bulgarian', 'English'],
            'grades_average': 5.5
        },
        {
            'math_interest': 4,
            'science_interest': 5,
            'art_interest': 9,
            'sports_interest': 6,
            'study_hours_per_day': 3,
            'preferred_subject': 'Art',
            'career_goal': 'Designer',
            'extracurricular': True,
            'leadership_experience': False,
            'team_preference': True,
            'languages_spoken': ['Bulgarian', 'English', 'French'],
            'grades_average': 5.2
        },
        {
            'math_interest': 9,
            'science_interest': 8,
            'art_interest': 2,
            'sports_interest': 4,
            'study_hours_per_day': 6,
            'preferred_subject': 'Physics',
            'career_goal': 'Scientist',
            'extracurricular': False,
            'leadership_experience': True,
            'team_preference': False,
            'languages_spoken': ['Bulgarian', 'English', 'German'],
            'grades_average': 5.8
        }
    ]
    
    
    print("Extracting features from survey responses...")
    survey_features = []
    for i, survey in enumerate(sample_surveys):
        features = extract_features_from_survey_response(survey)
        survey_features.append(features.flatten())
        print(f"Survey {i+1} features: {features.flatten()}")
    
    X_survey = np.array(survey_features)
    print(f"\nSurvey features shape: {X_survey.shape}")
    
    
    y_programs = np.array([0, 2, 1])  
    program_names = ['Engineering', 'Science', 'Arts']
    
    print(f"Target programs: {[program_names[y] for y in y_programs]}")
    
    
    print("\nGenerating synthetic dataset based on survey patterns...")
    n_synthetic = 300
    X_synthetic = []
    y_synthetic = []
    
    np.random.seed(42)
    for _ in range(n_synthetic):
        
        pattern = np.random.choice([0, 1, 2])
        base_survey = sample_surveys[pattern].copy()
        
        
        for key, value in base_survey.items():
            if isinstance(value, (int, float)) and key != 'grades_average':
                noise = np.random.normal(0, 0.5)
                base_survey[key] = max(1, min(10, value + noise))
            elif key == 'grades_average':
                noise = np.random.normal(0, 0.2)
                base_survey[key] = max(2.0, min(6.0, value + noise))
        
        features = extract_features_from_survey_response(base_survey)
        X_synthetic.append(features.flatten())
        y_synthetic.append(y_programs[pattern])
    
    X_synthetic = np.array(X_synthetic)
    y_synthetic = np.array(y_synthetic)
    
    print(f"Synthetic dataset shape: X={X_synthetic.shape}, y={y_synthetic.shape}")
    
    
    print("\nTraining model on survey data...")
    survey_pipeline = MLTrainingPipeline(random_seed=42)
    
    survey_results = survey_pipeline.train_model(
        X_synthetic, y_synthetic,
        hidden_sizes=[16, 8],
        activation='relu',
        learning_rate=0.01,
        epochs=100,
        batch_size=32,
        test_size=0.2,
        val_size=0.2,
        verbose=True
    )
    
    
    print("\nTesting on original survey responses...")
    predictions = survey_pipeline.predict(X_survey)
    class_predictions = survey_pipeline.predict_classes(X_survey)
    
    print("\nPrediction Results:")
    for i, (survey, pred_probs, pred_class) in enumerate(zip(sample_surveys, predictions, class_predictions)):
        print(f"\nSurvey {i+1}:")
        print(f"  Interests: Math={survey['math_interest']}, Science={survey['science_interest']}, Art={survey['art_interest']}")
        print(f"  Career Goal: {survey['career_goal']}")
        print(f"  Prediction probabilities: {pred_probs}")
        print(f"  Predicted program: {program_names[pred_class]} (confidence: {pred_probs[pred_class]:.3f})")
        print(f"  Actual program: {program_names[y_programs[i]]}")
    
    
    survey_model_path = "saved_models/survey_recommendation_model.pkl"
    survey_pipeline.save_pipeline(survey_model_path)
    print(f"\nSurvey model saved to: {survey_model_path}")
    
    return survey_pipeline, survey_results


def demo_model_comparison():
    """Demonstrate comparison between different model configurations."""
    print("\n" + "=" * 60)
    print("DEMO 4: Model Architecture Comparison")
    print("=" * 60)
    
    
    X, y = create_sample_dataset(n_samples=800, n_features=10, n_classes=3)
    
    
    architectures = [
        {"hidden_sizes": [16], "name": "Single Layer (16)"},
        {"hidden_sizes": [32, 16], "name": "Two Layer (32-16)"},
        {"hidden_sizes": [64, 32, 16], "name": "Three Layer (64-32-16)"},
        {"hidden_sizes": [128, 64, 32], "name": "Three Layer Large (128-64-32)"}
    ]
    
    results = []
    
    print("Comparing different neural network architectures...")
    for arch in architectures:
        print(f"\nTraining {arch['name']}...")
        
        pipeline = MLTrainingPipeline(random_seed=42)
        result = pipeline.train_model(
            X, y,
            hidden_sizes=arch['hidden_sizes'],
            activation='relu',
            learning_rate=0.01,
            epochs=50,
            batch_size=32,
            verbose=False  
        )
        
        metrics = result['test_metrics']
        results.append({
            'name': arch['name'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision_macro'],
            'recall': metrics['recall_macro'],
            'f1_score': metrics['f1_macro'],
            'architecture': arch['hidden_sizes']
        })
        
        print(f"Accuracy: {metrics['accuracy']:.4f}, F1-score: {metrics['f1_macro']:.4f}")
    
    
    print("\n" + "-" * 80)
    print("ARCHITECTURE COMPARISON RESULTS")
    print("-" * 80)
    print(f"{'Architecture':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<25} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} {result['f1_score']:<10.4f}")
    
    
    best_arch = max(results, key=lambda x: x['f1_score'])
    print(f"\nBest performing architecture: {best_arch['name']} with F1-score: {best_arch['f1_score']:.4f}")
    
    return results


def main():
    """Run all demonstrations."""
    print("CUSTOM ML MODEL DEVELOPMENT DEMONSTRATION")
    print("========================================")
    print("This script demonstrates a complete ML pipeline implemented from scratch.")
    print("Features demonstrated:")
    print("1. Custom Neural Network Implementation")
    print("2. Complete Training Pipeline")
    print("3. Comprehensive Model Evaluation")
    print("4. Model Persistence (Save/Load)")
    print("5. Survey Data Integration")
    print("6. Architecture Comparison")
    print("\nStarting demonstrations...\n")
    
    try:
        
        os.makedirs("saved_models", exist_ok=True)
        
        
        demo1_results = demo_basic_neural_network()
        demo2_results = demo_ml_pipeline()
        demo3_results = demo_survey_integration()
        demo4_results = demo_model_comparison()
        
        print("\n" + "=" * 60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Models and pipelines have been saved to the 'saved_models/' directory.")
        print("You can now integrate these components into your Flask application.")
        print("\nNext steps:")
        print("1. Integrate the survey model with your Flask app")
        print("2. Create endpoints for model training and prediction")
        print("3. Add visualization for model performance")
        print("4. Implement model retraining with new survey data")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 