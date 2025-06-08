import unittest
import time
import numpy as np
import psutil
import os
from contextlib import contextmanager

from app.ml.models import CustomNeuralNetwork
from app.ml.service import MLModelService
from app.ml.pipeline import MLTrainingPipeline


@contextmanager
def measure_time():
    """Context manager to measure execution time."""
    start_time = time.time()
    yield lambda: time.time() - start_time
    

@contextmanager
def measure_memory():
    """Context manager to measure memory usage."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield lambda: process.memory_info().rss / 1024 / 1024 - initial_memory


class TestMLPerformance(unittest.TestCase):
    """Performance and scalability tests for ML models."""
    
    def setUp(self):
        """Set up test environment."""
        np.random.seed(42)
        
    def test_training_performance_small_dataset(self):
        """Test training performance on small dataset."""
        X_train = np.random.randn(1000, 10)
        y_train = np.random.randint(0, 2, (1000, 1))
        
        model = CustomNeuralNetwork(
            input_size=10,
            hidden_sizes=[20, 10],
            output_size=1,
            learning_rate=0.01,
            random_seed=42
        )
        
        with measure_time() as get_time:
            with measure_memory() as get_memory:
                model.train(X_train, y_train, epochs=50, batch_size=32, verbose=False)
        
        training_time = get_time()
        memory_used = get_memory()
        
        print(f"Small dataset training time: {training_time:.2f} seconds")
        print(f"Memory used: {memory_used:.2f} MB")
        
        # Performance assertions
        self.assertLess(training_time, 30.0, "Training should complete within 30 seconds")
        self.assertLess(memory_used, 100.0, "Memory usage should be less than 100 MB")
        
    def test_training_performance_medium_dataset(self):
        """Test training performance on medium dataset."""
        X_train = np.random.randn(10000, 20)
        y_train = np.random.randint(0, 3, (10000, 3))  # One-hot encoded
        
        model = CustomNeuralNetwork(
            input_size=20,
            hidden_sizes=[50, 25],
            output_size=3,
            output_activation='softmax',
            learning_rate=0.01,
            random_seed=42
        )
        
        with measure_time() as get_time:
            with measure_memory() as get_memory:
                model.train(X_train, y_train, epochs=30, batch_size=64, verbose=False)
        
        training_time = get_time()
        memory_used = get_memory()
        
        print(f"Medium dataset training time: {training_time:.2f} seconds")
        print(f"Memory used: {memory_used:.2f} MB")
        
        # Performance assertions
        self.assertLess(training_time, 120.0, "Training should complete within 2 minutes")
        self.assertLess(memory_used, 500.0, "Memory usage should be less than 500 MB")
        
    def test_prediction_speed(self):
        """Test prediction speed for different batch sizes."""
        model = CustomNeuralNetwork(
            input_size=10,
            hidden_sizes=[20, 10],
            output_size=1,
            random_seed=42
        )
        
        # Train model quickly
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, (100, 1))
        model.train(X_train, y_train, epochs=5, verbose=False)
        
        batch_sizes = [1, 10, 100, 1000, 10000]
        
        for batch_size in batch_sizes:
            X_test = np.random.randn(batch_size, 10)
            
            with measure_time() as get_time:
                predictions = model.predict(X_test)
            
            prediction_time = get_time()
            predictions_per_second = batch_size / prediction_time if prediction_time > 0 else float('inf')
            
            print(f"Batch size {batch_size}: {prediction_time:.4f}s, {predictions_per_second:.0f} predictions/sec")
            
            # Ensure predictions are reasonable
            self.assertEqual(predictions.shape, (batch_size, 1))
            self.assertGreater(predictions_per_second, 100, 
                             f"Should predict at least 100 samples/sec for batch size {batch_size}")
            
    def test_memory_scaling(self):
        """Test memory usage scaling with model size."""
        input_size = 50
        layer_configs = [
            [10],           # Small model
            [50, 25],       # Medium model
            [100, 50, 25],  # Large model
        ]
        
        for i, hidden_sizes in enumerate(layer_configs):
            with measure_memory() as get_memory:
                model = CustomNeuralNetwork(
                    input_size=input_size,
                    hidden_sizes=hidden_sizes,
                    output_size=5,
                    random_seed=42
                )
                
                # Create some data and do a forward pass
                X = np.random.randn(100, input_size)
                predictions = model.predict(X)
                
            memory_used = get_memory()
            total_params = self._count_parameters(model)
            
            print(f"Model {i+1} ({hidden_sizes}): {total_params} parameters, {memory_used:.2f} MB")
            
            # Ensure memory usage is reasonable
            self.assertLess(memory_used, 50.0, "Memory usage should be reasonable")
            
    def test_batch_size_performance(self):
        """Test performance with different batch sizes."""
        X_train = np.random.randn(5000, 15)
        y_train = np.random.randint(0, 2, (5000, 1))
        
        batch_sizes = [16, 32, 64, 128, 256]
        performance_results = []
        
        for batch_size in batch_sizes:
            model = CustomNeuralNetwork(
                input_size=15,
                hidden_sizes=[30, 15],
                output_size=1,
                random_seed=42
            )
            
            with measure_time() as get_time:
                history = model.train(
                    X_train, y_train,
                    epochs=10,
                    batch_size=batch_size,
                    verbose=False
                )
            
            training_time = get_time()
            final_loss = history['loss'][-1]
            
            performance_results.append({
                'batch_size': batch_size,
                'time': training_time,
                'final_loss': final_loss
            })
            
            print(f"Batch size {batch_size}: {training_time:.2f}s, final loss: {final_loss:.4f}")
        
        # Ensure all batch sizes complete training
        for result in performance_results:
            self.assertLess(result['time'], 60.0, 
                           f"Batch size {result['batch_size']} should train within 60 seconds")
            self.assertIsNotNone(result['final_loss'])
            
    def test_concurrent_predictions(self):
        """Test concurrent prediction performance."""
        import threading
        import queue
        
        # Train a model
        model = CustomNeuralNetwork(
            input_size=10,
            hidden_sizes=[20],
            output_size=1,
            random_seed=42
        )
        
        X_train = np.random.randn(500, 10)
        y_train = np.random.randint(0, 2, (500, 1))
        model.train(X_train, y_train, epochs=10, verbose=False)
        
        def make_predictions(model, input_queue, result_queue):
            """Worker function for concurrent predictions."""
            while True:
                try:
                    X_test = input_queue.get(timeout=1)
                    if X_test is None:
                        break
                    
                    start_time = time.time()
                    predictions = model.predict(X_test)
                    end_time = time.time()
                    
                    result_queue.put({
                        'predictions': predictions,
                        'time': end_time - start_time
                    })
                    input_queue.task_done()
                except queue.Empty:
                    break
        
        # Test concurrent predictions
        input_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Add test data to queue
        for _ in range(10):
            X_test = np.random.randn(100, 10)
            input_queue.put(X_test)
        
        # Start worker threads
        num_threads = 3
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(
                target=make_predictions,
                args=(model, input_queue, result_queue)
            )
            thread.start()
            threads.append(thread)
        
        with measure_time() as get_time:
            input_queue.join()  # Wait for all tasks to complete
        
        # Signal threads to stop
        for _ in range(num_threads):
            input_queue.put(None)
        
        # Wait for threads to finish
        for thread in threads:
            thread.join()
        
        total_time = get_time()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        print(f"Concurrent predictions: {len(results)} batches in {total_time:.2f}s")
        
        # Ensure all predictions completed successfully
        self.assertEqual(len(results), 10, "All prediction batches should complete")
        
        for result in results:
            self.assertEqual(result['predictions'].shape, (100, 1))
            self.assertLess(result['time'], 1.0, "Individual predictions should be fast")
            
    def _count_parameters(self, model):
        """Count total number of parameters in the model."""
        total_params = 0
        for layer in model.layers:
            total_params += layer['weight'].size + layer['bias'].size
        return total_params


class TestMLStressTests(unittest.TestCase):
    """Stress tests for ML models."""
    
    def test_large_model_training(self):
        """Test training with a large model architecture."""
        # Skip if insufficient memory
        available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
        if available_memory < 2000:  # Less than 2GB
            self.skipTest("Insufficient memory for large model test")
            
        X_train = np.random.randn(5000, 100)
        y_train = np.random.randint(0, 10, (5000, 10))  # One-hot encoded
        
        model = CustomNeuralNetwork(
            input_size=100,
            hidden_sizes=[200, 100, 50],
            output_size=10,
            output_activation='softmax',
            learning_rate=0.001,
            random_seed=42
        )
        
        with measure_time() as get_time:
            with measure_memory() as get_memory:
                model.train(X_train, y_train, epochs=20, batch_size=64, verbose=False)
        
        training_time = get_time()
        memory_used = get_memory()
        
        print(f"Large model training: {training_time:.2f}s, {memory_used:.2f} MB")
        
        # Test predictions
        X_test = np.random.randn(1000, 100)
        predictions = model.predict(X_test)
        
        self.assertEqual(predictions.shape, (1000, 10))
        self.assertLess(training_time, 300.0, "Large model should train within 5 minutes")
        
    def test_repeated_training_sessions(self):
        """Test model stability over repeated training sessions."""
        accuracies = []
        
        for session in range(5):
            # Create fresh data for each session
            X_train = np.random.randn(1000, 20)
            y_train = (X_train.sum(axis=1) > 0).astype(int).reshape(-1, 1)
            
            X_test = np.random.randn(200, 20)
            y_test = (X_test.sum(axis=1) > 0).astype(int).reshape(-1, 1)
            
            model = CustomNeuralNetwork(
                input_size=20,
                hidden_sizes=[40, 20],
                output_size=1,
                random_seed=42 + session  # Different seed each time
            )
            
            model.train(X_train, y_train, epochs=30, verbose=False)
            
            # Test accuracy
            predictions = model.predict(X_test)
            accuracy = model.calculate_accuracy(y_test.flatten(), predictions.flatten())
            accuracies.append(accuracy)
            
            print(f"Session {session + 1}: Accuracy = {accuracy:.3f}")
        
        # Check consistency
        self.assertEqual(len(accuracies), 5)
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f"Mean accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        
        # Ensure reasonable performance and consistency
        self.assertGreater(mean_accuracy, 0.45, "Mean accuracy should be reasonable")
        self.assertLess(std_accuracy, 0.1, "Accuracy should be consistent across sessions")


class TestMLBenchmarks(unittest.TestCase):
    """Benchmark tests comparing different configurations."""
    
    def test_activation_function_performance(self):
        """Compare performance of different activation functions."""
        activations = ['sigmoid', 'relu', 'tanh']
        results = {}
        
        for activation in activations:
            model = CustomNeuralNetwork(
                input_size=20,
                hidden_sizes=[40, 20],
                output_size=1,
                activation=activation,
                random_seed=42
            )
            
            X_train = np.random.randn(2000, 20)
            y_train = np.random.randint(0, 2, (2000, 1))
            
            with measure_time() as get_time:
                history = model.train(X_train, y_train, epochs=20, verbose=False)
            
            training_time = get_time()
            final_loss = history['loss'][-1]
            
            results[activation] = {
                'time': training_time,
                'final_loss': final_loss
            }
            
            print(f"{activation}: {training_time:.2f}s, final loss: {final_loss:.4f}")
        
        # Ensure all activations work
        for activation, result in results.items():
            self.assertLess(result['time'], 60.0, f"{activation} should train within 60s")
            self.assertIsNotNone(result['final_loss'])
            
    def test_model_size_vs_performance(self):
        """Compare performance vs accuracy trade-offs for different model sizes."""
        architectures = [
            [10],              # Small
            [20, 10],          # Medium
            [40, 20, 10],      # Large
            [80, 40, 20, 10],  # Very large
        ]
        
        results = []
        
        for i, hidden_sizes in enumerate(architectures):
            model = CustomNeuralNetwork(
                input_size=15,
                hidden_sizes=hidden_sizes,
                output_size=1,
                random_seed=42
            )
            
            # Create consistent dataset
            np.random.seed(42)
            X_train = np.random.randn(3000, 15)
            y_train = (X_train.sum(axis=1) > 0).astype(int).reshape(-1, 1)
            
            X_test = np.random.randn(500, 15)
            y_test = (X_test.sum(axis=1) > 0).astype(int).reshape(-1, 1)
            
            with measure_time() as get_time:
                model.train(X_train, y_train, epochs=25, verbose=False)
            
            training_time = get_time()
            
            # Test accuracy
            predictions = model.predict(X_test)
            accuracy = model.calculate_accuracy(y_test.flatten(), predictions.flatten())
            
            param_count = sum(layer['weight'].size + layer['bias'].size for layer in model.layers)
            
            results.append({
                'architecture': hidden_sizes,
                'parameters': param_count,
                'training_time': training_time,
                'accuracy': accuracy
            })
            
            print(f"Architecture {hidden_sizes}: {param_count} params, "
                  f"{training_time:.2f}s, {accuracy:.3f} accuracy")
        
        # Check that larger models don't take unreasonably long
        for result in results:
            self.assertLess(result['training_time'], 120.0, 
                           f"Architecture {result['architecture']} should train within 2 minutes")
            self.assertGreater(result['accuracy'], 0.4, 
                             f"Architecture {result['architecture']} should achieve reasonable accuracy")


if __name__ == '__main__':
    # Run with more verbose output for performance tests
    unittest.main(verbosity=2) 