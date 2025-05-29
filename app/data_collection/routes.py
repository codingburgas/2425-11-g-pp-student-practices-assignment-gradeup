"""
Data Collection Routes

Handles the API endpoints for survey data collection.
"""

from flask import request, jsonify, render_template
from . import data_collection
from .validators import SurveyValidator, ResponseValidator

@data_collection.route('/surveys', methods=['GET'])
def get_surveys():
    """Retrieve list of available surveys."""
    # Basic endpoint setup - will be enhanced in later commits
    return jsonify({
        'surveys': [],
        'message': 'Data collection system initialized'
    })

@data_collection.route('/surveys/submit', methods=['POST'])
def submit_survey():
    """Submit survey response."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'valid': False
            }), 400
        
        # Basic validation schema for demo
        demo_schema = {
            'required_fields': ['name', 'email'],
            'fields': {
                'name': {'type': 'text', 'min_length': 2, 'max_length': 100},
                'email': {'type': 'email'},
                'rating': {'type': 'rating', 'min': 1, 'max': 5},
                'phone': {'type': 'phone'}
            }
        }
        
        validator = ResponseValidator(demo_schema)
        validation_result = validator.validate_response(data)
        
        if validation_result['valid']:
            return jsonify({
                'message': 'Survey data validated successfully',
                'valid': True,
                'status': 'validated'
            })
        else:
            return jsonify({
                'message': 'Validation failed',
                'valid': False,
                'errors': validation_result['errors']
            }), 400
            
    except Exception as e:
        return jsonify({
            'error': 'Internal server error during validation',
            'valid': False
        }), 500

@data_collection.route('/surveys/validate', methods=['POST'])
def validate_survey_data():
    """Endpoint to validate survey data without submitting."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'valid': False
            }), 400
        
        # Use individual validators for quick checks
        validator = SurveyValidator()
        validation_results = {}
        
        if 'email' in data:
            validation_results['email'] = validator.validate_email(data['email'])
        
        if 'phone' in data:
            validation_results['phone'] = validator.validate_phone(data['phone'])
        
        if 'rating' in data:
            validation_results['rating'] = validator.validate_rating_scale(data['rating'])
        
        return jsonify({
            'validation_results': validation_results,
            'message': 'Individual field validation completed'
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Validation error',
            'valid': False
        }), 500 