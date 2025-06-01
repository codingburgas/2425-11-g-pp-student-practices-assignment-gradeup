"""
ML Blueprint Module

This module contains the Flask blueprint for ML endpoints.
"""

from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for, current_app
from flask_login import login_required, current_user
import os

from .service import MLModelService


ml_bp = Blueprint('ml', __name__, url_prefix='/ml')


ml_service = MLModelService()


def initialize_ml_service(app):
    """Initialize the ML service when the app starts."""
    try:
        ml_service.initialize(app.instance_path)
        ml_service.load_model()
        app.logger.info("ML service initialized successfully")
    except Exception as e:
        app.logger.error(f"Error initializing ML service: {e}")



@ml_bp.record_once
def on_register(state):
    """Called when the blueprint is registered with the app."""
    initialize_ml_service(state.app)


@ml_bp.route('/train', methods=['GET', 'POST'])
@login_required
def train_model():
    """Train the ML model with current survey data."""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        force_retrain = request.form.get('force_retrain', False, type=bool)
        
        
        survey_responses = []  
        
        
        result = ml_service.train_model(survey_responses, force_retrain=force_retrain)
        
        if result['status'].startswith('Training completed'):
            flash(f"Model training completed successfully! Accuracy: {result.get('test_accuracy', 0):.3f}", 'success')
        else:
            flash(f"Training failed: {result['status']}", 'error')
        
        return redirect(url_for('ml.model_status'))
    
    
    survey_count = 0  
    return render_template('ml/train.html', survey_count=survey_count)


@ml_bp.route('/predict', methods=['POST'])
@login_required
def predict_programs():
    """API endpoint for getting program predictions."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        
        predictions = ml_service.predict_programs(data, top_k=5)
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions)
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in predict endpoint: {e}")
        return jsonify({'error': 'Prediction failed'}), 500


@ml_bp.route('/recommend/<int:survey_response_id>')
@login_required
def get_recommendations(survey_response_id):
    """Get recommendations for a specific survey response."""
    
    
    
    
    survey_response = {
        'id': survey_response_id,
        'user_id': current_user.id
    }
    
    
    answers = {
        'math_interest': 7,
        'science_interest': 8,
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
    }
    
    
    predictions = ml_service.predict_programs(answers, top_k=10)
    
    
    recommendations = []
    for pred in predictions:
        recommendations.append({
            'program': {
                'id': pred['program_id'],
                'name': pred['program_name'],
                'description': f"A comprehensive program in {pred['program_name']}"
            },
            'school': {
                'name': pred['school_name']
            },
            'confidence': pred['confidence'],
            'rank': pred['rank']
        })
    
    return render_template('ml/recommendations.html', 
                         survey_response=survey_response,
                         recommendations=recommendations)


@ml_bp.route('/status')
@login_required
def model_status():
    """Show ML model status and statistics."""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('main.index'))
    
    
    if not ml_service.is_trained:
        ml_service.load_model()
    
    
    survey_count = 0
    program_count = 4  
    recommendation_count = 0
    
    model_info = {
        'is_trained': ml_service.is_trained,
        'model_path': ml_service.model_path,
        'model_exists': os.path.exists(ml_service.model_path) if ml_service.model_path else False
    }
    
    stats = {
        'surveys': survey_count,
        'programs': program_count,
        'recommendations': recommendation_count
    }
    
    return render_template('ml/status.html', model_info=model_info, stats=stats)


@ml_bp.route('/test', methods=['GET', 'POST'])
@login_required
def test_model():
    """Test the ML model with sample data."""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        
        test_data = {
            'math_interest': request.form.get('math_interest', type=int),
            'science_interest': request.form.get('science_interest', type=int),
            'art_interest': request.form.get('art_interest', type=int),
            'sports_interest': request.form.get('sports_interest', type=int),
            'study_hours_per_day': request.form.get('study_hours_per_day', type=int),
            'preferred_subject': request.form.get('preferred_subject'),
            'career_goal': request.form.get('career_goal'),
            'extracurricular': request.form.get('extracurricular') == 'true',
            'leadership_experience': request.form.get('leadership_experience') == 'true',
            'team_preference': request.form.get('team_preference') == 'true',
            'languages_spoken': request.form.get('languages_spoken', '').split(','),
            'grades_average': request.form.get('grades_average', type=float)
        }
        
        
        predictions = ml_service.predict_programs(test_data, top_k=5)
        
        
        recommendations = []
        for pred in predictions:
            recommendations.append({
                'program': {
                    'name': pred['program_name'],
                    'description': f"A comprehensive program in {pred['program_name']}"
                },
                'school': {
                    'name': pred['school_name']
                },
                'confidence': pred['confidence'],
                'rank': pred['rank']
            })
        
        return render_template('ml/test.html', 
                             test_data=test_data,
                             recommendations=recommendations)
    
    return render_template('ml/test.html') 