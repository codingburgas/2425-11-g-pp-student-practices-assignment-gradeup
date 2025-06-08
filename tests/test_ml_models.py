import unittest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from flask import Flask

from app.ml.models import CustomNeuralNetwork, ActivationFunctions, LossFunctions
from app.ml.service import MLModelService
from app.ml.evaluator import ModelEvaluator
from app.ml.pipeline import MLTrainingPipeline
from app.ml.utils import extract_features_from_survey_response


class TestActivationFunctions(unittest.TestCase):
    """Test all activation functions and their derivatives."""
    
    def setUp(self):
        self.test_input = np.array([[1.0, -1.0, 0.0, 2.0]])
        
    def test_sigmoid(self):
        """Test sigmoid activation function."""
        result = ActivationFunctions.sigmoid(self.test_input)
        
        # Check shape
        self.assertEqual(result.shape, self.test_input.shape)
        
        # Check values are in range [0, 1]
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        
        # Check specific values
        expected_0 = 1 / (1 + np.exp(-0))
        self.assertAlmostEqual(result[0, 2], expected_0, places=5)
        
    def test_sigmoid_derivative(self):
        """Test sigmoid derivative."""
        result = ActivationFunctions.sigmoid_derivative(self.test_input)
        self.assertEqual(result.shape, self.test_input.shape)
        self.assertTrue(np.all(result >= 0))
        
    def test_relu(self):
        """Test ReLU activation function."""
        result = ActivationFunctions.relu(self.test_input)
        
        expected = np.array([[1.0, 0.0, 0.0, 2.0]])
        np.testing.assert_array_equal(result, expected)
        
    def test_relu_derivative(self):
        """Test ReLU derivative."""
        result = ActivationFunctions.relu_derivative(self.test_input)
        expected = np.array([[1.0, 0.0, 0.0, 1.0]])
        np.testing.assert_array_equal(result, expected)
        
    def test_tanh(self):
        """Test tanh activation function."""
        result = ActivationFunctions.tanh(self.test_input)
        
        # Check values are in range [-1, 1]
        self.assertTrue(np.all(result >= -1))
        self.assertTrue(np.all(result <= 1))
        
        # Check zero input gives zero output
        self.assertAlmostEqual(result[0, 2], 0.0, places=5)
        
    def test_softmax(self):
        """Test softmax activation function."""
        test_input_2d = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        result = ActivationFunctions.softmax(test_input_2d)
        
        # Check shape
        self.assertEqual(result.shape, test_input_2d.shape)
        
        # Check probabilities sum to 1 for each row
        for i in range(result.shape[0]):
            self.assertAlmostEqual(np.sum(result[i, :]), 1.0, places=5)


class TestLossFunctions(unittest.TestCase):
    """Test loss functions."""
    
    def setUp(self):
        self.y_true = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        self.y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        
    def test_mean_squared_error(self):
        """Test MSE loss function."""
        result = LossFunctions.mean_squared_error(self.y_true, self.y_pred)
        
        self.assertIsInstance(result, (float, np.float64))
        self.assertGreaterEqual(result, 0)
        
    def test_mse_derivative(self):
        """Test MSE derivative."""
        result = LossFunctions.mse_derivative(self.y_true, self.y_pred)
        
        self.assertEqual(result.shape, self.y_true.shape)
        
    def test_binary_crossentropy(self):
        """Test binary crossentropy loss."""
        y_true_binary = np.array([1, 0, 1])
        y_pred_binary = np.array([0.9, 0.1, 0.7])
        
        result = LossFunctions.binary_crossentropy(y_true_binary, y_pred_binary)
        
        self.assertIsInstance(result, (float, np.float64))
        self.assertGreaterEqual(result, 0)
        
    def test_categorical_crossentropy(self):
        """Test categorical crossentropy loss."""
        result = LossFunctions.categorical_crossentropy(self.y_true, self.y_pred)
        
        self.assertIsInstance(result, (float, np.float64))
        self.assertGreaterEqual(result, 0)


class TestCustomNeuralNetwork(unittest.TestCase):
    """Test the custom neural network implementation."""
    
    def setUp(self):
        """Set up test data and model."""
        np.random.seed(42)
        
        # Create simple test data
        self.X_train = np.random.randn(100, 4)
        self.y_train = (self.X_train[:, 0] + self.X_train[:, 1] > 0).astype(int).reshape(-1, 1)
        
        self.X_test = np.random.randn(20, 4)
        self.y_test = (self.X_test[:, 0] + self.X_test[:, 1] > 0).astype(int).reshape(-1, 1)
        
        # Create model
        self.model = CustomNeuralNetwork(
            input_size=4,
            hidden_sizes=[8, 4],
            output_size=1,
            activation='relu',
            learning_rate=0.01,
            random_seed=42
        )
        
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.input_size, 4)
        self.assertEqual(self.model.hidden_sizes, [8, 4])
        self.assertEqual(self.model.output_size, 1)
        self.assertEqual(len(self.model.layers), 3)  # 2 hidden + 1 output
        
        # Check layer dimensions
        self.assertEqual(self.model.layers[0]['weight'].shape, (4, 8))
        self.assertEqual(self.model.layers[1]['weight'].shape, (8, 4))
        self.assertEqual(self.model.layers[2]['weight'].shape, (4, 1))
        
    def test_forward_propagation(self):
        """Test forward propagation."""
        predictions = self.model.forward_propagation(self.X_test)
        
        # Check output shape
        self.assertEqual(predictions.shape, (20, 1))
        
        # Check output range for sigmoid
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
        
    def test_predict(self):
        """Test prediction method."""
        predictions = self.model.predict(self.X_test)
        
        self.assertEqual(predictions.shape, (20, 1))
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
        
    def test_predict_classes(self):
        """Test class prediction method."""
        class_predictions = self.model.predict_classes(self.X_test)
        
        self.assertEqual(class_predictions.shape, (20,))
        self.assertTrue(np.all(np.isin(class_predictions, [0, 1])))
        
    def test_training(self):
        """Test model training."""
        initial_predictions = self.model.predict(self.X_test)
        
        # Train the model
        history = self.model.train(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            epochs=10,
            batch_size=16,
            verbose=False
        )
        
        # Check training history
        self.assertIn('loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['loss']), 10)
        
        # Check if model learned something (predictions changed)
        final_predictions = self.model.predict(self.X_test)
        self.assertFalse(np.array_equal(initial_predictions, final_predictions))
        
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        
        accuracy = self.model.calculate_accuracy(y_true, y_pred)
        
        self.assertIsInstance(accuracy, (float, np.float64))
        self.assertGreater(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        
    def test_model_save_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.pkl')
            
            # Train model a bit first
            self.model.train(self.X_train, self.y_train, epochs=5, verbose=False)
            original_predictions = self.model.predict(self.X_test)
            
            # Save model
            self.model.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load model
            loaded_model = CustomNeuralNetwork.load_model(model_path)
            loaded_predictions = loaded_model.predict(self.X_test)
            
            # Check predictions are the same
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=5)


class TestModelEvaluator(unittest.TestCase):
    """Test model evaluation metrics."""
    
    def setUp(self):
        self.y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        self.y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1])
        self.num_classes = 3
        
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        cm = ModelEvaluator.confusion_matrix(self.y_true, self.y_pred, self.num_classes)
        
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(np.sum(cm), len(self.y_true))
        
    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 score calculation."""
        precision, recall, f1 = ModelEvaluator.precision_recall_f1(
            self.y_true, self.y_pred, self.num_classes, 'macro'
        )
        
        self.assertIsInstance(precision, (float, np.float64))
        self.assertIsInstance(recall, (float, np.float64))
        self.assertIsInstance(f1, (float, np.float64))
        
        self.assertGreaterEqual(precision, 0)
        self.assertGreaterEqual(recall, 0)
        self.assertGreaterEqual(f1, 0)
        
    def test_evaluate_model(self):
        """Test comprehensive model evaluation."""
        # Create a simple trained model
        X_test = np.random.randn(50, 4)
        y_test = np.random.randint(0, 3, (50, 3))  # One-hot encoded
        
        model = CustomNeuralNetwork(
            input_size=4,
            hidden_sizes=[8],
            output_size=3,
            output_activation='softmax',
            random_seed=42
        )
        
        metrics = ModelEvaluator.evaluate_model(model, X_test, y_test)
        
        # Check all expected metrics are present
        expected_metrics = [
            'accuracy', 'confusion_matrix', 'precision_macro', 'recall_macro', 'f1_macro',
            'precision_micro', 'recall_micro', 'f1_micro',
            'precision_weighted', 'recall_weighted', 'f1_weighted',
            'predictions', 'prediction_probabilities'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)


class TestMLModelService(unittest.TestCase):
    """Test the ML model service layer."""
    
    def setUp(self):
        self.service = MLModelService()
        # Create a test Flask app for context
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app_context = self.app.app_context()
        self.app_context.push()
        
    def tearDown(self):
        self.app_context.pop()
        
    def test_service_initialization(self):
        """Test service initialization."""
        instance_path = '/tmp/test'
        self.service.initialize(instance_path)
        
        expected_path = os.path.join(instance_path, 'models', 'recommendation_model.pkl')
        self.assertEqual(self.service.model_path, expected_path)
        
    @patch('app.ml.service.os.path.exists')
    @patch('app.ml.service.MLTrainingPipeline.load_pipeline')
    def test_load_model_success(self, mock_load, mock_exists):
        """Test successful model loading."""
        mock_exists.return_value = True
        mock_load.return_value = Mock()
        
        self.service.model_path = '/tmp/test_model.pkl'
        result = self.service.load_model()
        
        self.assertTrue(result)
        self.assertTrue(self.service.is_trained)
        mock_load.assert_called_once_with('/tmp/test_model.pkl')
        
    @patch('app.ml.service.os.path.exists')
    def test_load_model_file_not_found(self, mock_exists):
        """Test model loading when file doesn't exist."""
        mock_exists.return_value = False
        
        self.service.model_path = '/tmp/nonexistent_model.pkl'
        result = self.service.load_model()
        
        self.assertFalse(result)
        self.assertFalse(self.service.is_trained)
        
    def test_predict_programs_no_model(self):
        """Test prediction when no model is loaded."""
        survey_data = {"question_1": "answer_1"}
        result = self.service.predict_programs(survey_data)
        
        self.assertEqual(result, [])
        
    def test_prepare_training_data(self):
        """Test training data preparation."""
        # Mock survey responses
        mock_responses = [
            Mock(id=1, answers='{"1": "Science", "2": "Math"}'),
            Mock(id=2, answers='{"1": "Art", "2": "History"}')
        ]
        
        X, y, program_mapping = self.service._prepare_training_data(mock_responses)
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(program_mapping, dict)


class TestMLIntegration(unittest.TestCase):
    """Integration tests for ML components."""
    
    def test_end_to_end_training_prediction(self):
        """Test complete training and prediction pipeline."""
        # Create synthetic training data with proper label format
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        # Create proper labels (0, 1, 2) that will work with label encoding
        y_train = np.random.randint(0, 3, (100,))  # 1D array of class indices
        
        # Create and train pipeline
        pipeline = MLTrainingPipeline(random_seed=42)
        
        results = pipeline.train_model(
            X_train, y_train,
            hidden_sizes=[10, 5],
            epochs=10,
            batch_size=16,
            verbose=False
        )
        
        # Check training results
        self.assertIn('model', results)
        self.assertIn('test_metrics', results)
        self.assertIn('training_history', results)
        
        # Test predictions
        X_new = np.random.randn(5, 5)
        predictions = pipeline.predict(X_new)
        class_predictions = pipeline.predict_classes(X_new)
        
        self.assertEqual(predictions.shape, (5, 3))  # 3 classes
        self.assertEqual(class_predictions.shape, (5,))
        
    def test_feature_extraction(self):
        """Test feature extraction from survey responses."""
        survey_data = {
            "question_1": "Computer Science",
            "question_2": "5",
            "question_3": ["Math", "Science"],
            "question_4": "Urban"
        }
        
        features = extract_features_from_survey_response(survey_data)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features.shape), 2)  # Should be 2D array
        self.assertEqual(features.shape[0], 1)  # Single sample


if __name__ == '__main__':
    unittest.main() 