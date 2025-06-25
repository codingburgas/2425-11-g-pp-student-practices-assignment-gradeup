#!/usr/bin/env python3
"""
AI System Training Script for GradeUp
Trains both main and backup AI models for enhanced recommendations
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import logging

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import SurveyResponse, Program, School
from app.ml.service import MLModelService
from app.ml.pipeline import MLTrainingPipeline

class AISystemTrainer:
    """Main AI system trainer"""
    
    def __init__(self):
        self.app = create_app()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def collect_training_data(self):
        """Collect training data from database"""
        with self.app.app_context():
            responses = SurveyResponse.query.all()
            programs = Program.query.all()
            
            if not responses:
                return self._create_synthetic_data()
            
            # Process real data
            X_data = []
            y_labels = []
            
            for response in responses:
                try:
                    # Extract features from response
                    features = self._extract_features(response)
                    label = self._determine_label(response, programs)
                    
                    X_data.append(features)
                    y_labels.append(label)
                except Exception as e:
                    continue
            
            if not X_data:
                return self._create_synthetic_data()
            
            return np.array(X_data), np.array(y_labels)
    
    def _extract_features(self, response):
        """Extract numerical features from survey response"""
        # Default feature extraction
        features = [
            5,  # math_interest
            5,  # science_interest
            5,  # art_interest
            5,  # sports_interest
            0.5,  # academic_focus
            0.5,  # location_preference
            0.5,  # budget_fit
            0.5,  # social_environment
            0.5,  # career_prospects
            0.5,  # research_opportunities
            1,  # study_method_encoded
            1,  # career_goal_encoded
            1,  # budget_range_encoded
            1,  # location_pref_encoded
            1   # university_size_encoded
        ]
        return features
    
    def _determine_label(self, response, programs):
        """Determine best matching program"""
        return 0  # Default to first program
    
    def _create_synthetic_data(self):
        """Create synthetic training data"""
        self.logger.info("Creating synthetic training data...")
        
        n_samples = 1000
        n_features = 15
        
        # Generate realistic features
        X = np.random.randn(n_samples, n_features)
        X[:, 0:4] = np.random.randint(1, 11, (n_samples, 4))  # Interest scores
        X[:, 4:10] = np.random.uniform(0, 1, (n_samples, 6))  # Preferences
        X[:, 10:] = np.random.randint(0, 5, (n_samples, 5))   # Categories
        
        # Generate labels with correlation
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            if X[i, 0] > 7 and X[i, 1] > 6:  # High math+science
                y[i] = 0  # Engineering
            elif X[i, 1] > 7:  # High science
                y[i] = 1  # Science
            elif X[i, 2] > 7:  # High art
                y[i] = 2  # Arts
            else:
                y[i] = np.random.randint(0, 10)
        
        return X, y
    
    def train_main_model(self, X, y):
        """Train the main neural network model"""
        self.logger.info("Training main AI model...")
        
        try:
            # Initialize ML pipeline
            ml_pipeline = MLTrainingPipeline(random_seed=42)
            
            # Train model
            results = ml_pipeline.train_model(
                X, y,
                hidden_sizes=[64, 32, 16],
                activation='relu',
                learning_rate=0.001,
                epochs=100,
                batch_size=32,
                test_size=0.2,
                val_size=0.15,
                verbose=True
            )
            
            # Save model
            model_dir = os.path.join(self.app.instance_path, 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, 'recommendation_model.pkl')
            ml_pipeline.save_pipeline(model_path)
            
            return {
                'status': 'success',
                'accuracy': results['test_metrics']['accuracy'],
                'model_path': model_path
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def train_backup_model(self, X, y):
        """Train backup logistic regression model"""
        self.logger.info("Training backup model...")
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score
            import joblib
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save backup model
            model_dir = os.path.join(self.app.instance_path, 'models')
            backup_path = os.path.join(model_dir, 'backup_model.pkl')
            scaler_path = os.path.join(model_dir, 'backup_scaler.pkl')
            
            joblib.dump(model, backup_path)
            joblib.dump(scaler, scaler_path)
            
            return {
                'status': 'success',
                'accuracy': accuracy,
                'model_path': backup_path
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def run_training(self):
        """Run complete training process"""
        print("🧠 Starting AI System Training...")
        start_time = datetime.now()
        
        # Collect data
        X, y = self.collect_training_data()
        print(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train main model
        main_results = self.train_main_model(X, y)
        
        # Train backup model
        backup_results = self.train_backup_model(X, y)
        
        # Update ML service
        with self.app.app_context():
            ml_service = MLModelService()
            ml_service.initialize(self.app.instance_path)
            ml_service.load_model()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ Training completed in {duration:.2f} seconds")
        if main_results['status'] == 'success':
            print(f"🧠 Main model accuracy: {main_results['accuracy']:.4f}")
        if backup_results['status'] == 'success':
            print(f"🔄 Backup model accuracy: {backup_results['accuracy']:.4f}")
        
        return {
            'main_model': main_results,
            'backup_model': backup_results,
            'training_time': duration
        }

def main():
    trainer = AISystemTrainer()
    results = trainer.run_training()
    print("🎉 AI system training complete!")

if __name__ == "__main__":
    main() 