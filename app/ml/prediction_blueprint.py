"""
Flask Blueprint for Advanced Prediction System API

Provides REST API endpoints for:
- Individual predictions with confidence scoring
- Batch prediction processing
- Prediction history management
- Pattern analysis
"""

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from datetime import datetime
import json

from app import db
from app.models import User, SurveyResponse, PredictionHistory
from .prediction_system import advanced_prediction_system

# Create blueprint
prediction_bp = Blueprint('prediction', __name__, url_prefix='/api/prediction')


@prediction_bp.route('/predict', methods=['POST'])
@login_required
def predict_with_confidence():
    """
    Generate predictions with advanced confidence scoring.
    
    Expected JSON payload:
    {
        "survey_data": {...},
        "survey_response_id": 123,  // optional
        "top_k": 5,                 // optional, default 5
        "store_history": true       // optional, default true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'survey_data' not in data:
            return jsonify({
                'error': 'Missing required field: survey_data',
                'status': 'error'
            }), 400
        
        survey_data = data['survey_data']
        survey_response_id = data.get('survey_response_id')
        top_k = data.get('top_k', 5)
        store_history = data.get('store_history', True)
        
        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            return jsonify({
                'error': 'top_k must be an integer between 1 and 20',
                'status': 'error'
            }), 400
        
        # Generate predictions
        result = advanced_prediction_system.predict_with_confidence(
            survey_data=survey_data,
            user_id=current_user.id,
            survey_response_id=survey_response_id,
            top_k=top_k,
            store_history=store_history
        )
        
        # Check for errors in result
        if 'error' in result:
            return jsonify({
                'error': result['error'],
                'status': 'error',
                'result': result
            }), 500
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in predict_with_confidence endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500


@prediction_bp.route('/batch-predict', methods=['POST'])
@login_required
def batch_predict():
    """
    Process multiple predictions in batch.
    
    Expected JSON payload:
    {
        "prediction_requests": [
            {
                "survey_data": {...},
                "user_id": 123,              // optional, defaults to current user
                "survey_response_id": 456,   // optional
                "top_k": 5                   // optional
            },
            // ... more requests
        ],
        "store_history": true  // optional, default true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'prediction_requests' not in data:
            return jsonify({
                'error': 'Missing required field: prediction_requests',
                'status': 'error'
            }), 400
        
        prediction_requests = data['prediction_requests']
        store_history = data.get('store_history', True)
        
        if not isinstance(prediction_requests, list) or len(prediction_requests) == 0:
            return jsonify({
                'error': 'prediction_requests must be a non-empty list',
                'status': 'error'
            }), 400
        
        if len(prediction_requests) > 50:  # Limit batch size
            return jsonify({
                'error': 'Maximum batch size is 50 requests',
                'status': 'error'
            }), 400
        
        # Set default user_id to current user if not specified
        for request_item in prediction_requests:
            if 'user_id' not in request_item:
                request_item['user_id'] = current_user.id
            
            # Validate user has permission to predict for this user
            if request_item['user_id'] != current_user.id and not current_user.is_admin:
                return jsonify({
                    'error': 'Insufficient permissions to predict for other users',
                    'status': 'error'
                }), 403
        
        # Process batch predictions
        results = advanced_prediction_system.batch_predict(
            prediction_requests=prediction_requests,
            store_history=store_history
        )
        
        # Calculate summary statistics
        successful_count = len([r for r in results if r.get('predictions')])
        total_predictions = sum(len(r.get('predictions', [])) for r in results)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'summary': {
                'total_requests': len(prediction_requests),
                'successful_requests': successful_count,
                'total_predictions_generated': total_predictions,
                'timestamp': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in batch_predict endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500


@prediction_bp.route('/history', methods=['GET'])
@login_required
def get_prediction_history():
    """
    Retrieve prediction history for the current user.
    
    Query parameters:
    - limit: Maximum number of records (default: 10, max: 100)
    - prediction_type: Filter by type ('individual' or 'batch')
    - user_id: ID of user (admin only)
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', 10, type=int)
        prediction_type = request.args.get('prediction_type')
        user_id = request.args.get('user_id', current_user.id, type=int)
        
        # Validate limit
        if limit < 1 or limit > 100:
            return jsonify({
                'error': 'limit must be between 1 and 100',
                'status': 'error'
            }), 400
        
        # Check permissions
        if user_id != current_user.id and not current_user.is_admin:
            return jsonify({
                'error': 'Insufficient permissions to view other users\' history',
                'status': 'error'
            }), 403
        
        # Validate prediction_type
        if prediction_type and prediction_type not in ['individual', 'batch']:
            return jsonify({
                'error': 'prediction_type must be "individual" or "batch"',
                'status': 'error'
            }), 400
        
        # Get prediction history
        history = advanced_prediction_system.get_prediction_history(
            user_id=user_id,
            limit=limit,
            prediction_type=prediction_type
        )
        
        return jsonify({
            'status': 'success',
            'history': history,
            'count': len(history),
            'user_id': user_id
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in get_prediction_history endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500


@prediction_bp.route('/analyze-patterns', methods=['GET'])
@login_required
def analyze_prediction_patterns():
    """
    Analyze prediction patterns and trends for a user.
    
    Query parameters:
    - user_id: ID of user (admin only, defaults to current user)
    """
    try:
        user_id = request.args.get('user_id', current_user.id, type=int)
        
        # Check permissions
        if user_id != current_user.id and not current_user.is_admin:
            return jsonify({
                'error': 'Insufficient permissions to analyze other users\' patterns',
                'status': 'error'
            }), 403
        
        # Analyze patterns
        analysis = advanced_prediction_system.analyze_prediction_patterns(user_id)
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'user_id': user_id
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in analyze_prediction_patterns endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500


@prediction_bp.route('/confidence-threshold', methods=['GET', 'PUT'])
@login_required
def manage_confidence_threshold():
    """
    Get or update confidence threshold for predictions.
    
    GET: Returns current threshold
    PUT: Updates threshold (admin only)
    
    PUT payload:
    {
        "threshold": 0.4
    }
    """
    try:
        if request.method == 'GET':
            return jsonify({
                'status': 'success',
                'confidence_threshold': advanced_prediction_system.confidence_threshold
            })
        
        elif request.method == 'PUT':
            if not current_user.is_admin:
                return jsonify({
                    'error': 'Admin privileges required',
                    'status': 'error'
                }), 403
            
            data = request.get_json()
            if not data or 'threshold' not in data:
                return jsonify({
                    'error': 'Missing required field: threshold',
                    'status': 'error'
                }), 400
            
            threshold = data['threshold']
            
            # Validate threshold
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                return jsonify({
                    'error': 'threshold must be a number between 0 and 1',
                    'status': 'error'
                }), 400
            
            # Update threshold
            advanced_prediction_system.confidence_threshold = threshold
            
            return jsonify({
                'status': 'success',
                'message': 'Confidence threshold updated',
                'new_threshold': threshold
            })
    
    except Exception as e:
        current_app.logger.error(f"Error in manage_confidence_threshold endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500


@prediction_bp.route('/system-info', methods=['GET'])
@login_required
def get_system_info():
    """
    Get information about the prediction system.
    """
    try:
        # Get basic system information
        info = {
            'model_version': advanced_prediction_system.model_version,
            'confidence_threshold': advanced_prediction_system.confidence_threshold,
            'ml_service_initialized': advanced_prediction_system.ml_service.is_trained,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add admin-only information
        if current_user.is_admin:
            # Count total prediction history records
            total_predictions = PredictionHistory.query.count()
            
            # Count predictions by type
            individual_count = PredictionHistory.query.filter_by(prediction_type='individual').count()
            batch_count = PredictionHistory.query.filter_by(prediction_type='batch').count()
            
            # Count unique users with predictions
            unique_users = db.session.query(PredictionHistory.user_id).distinct().count()
            
            info.update({
                'total_prediction_records': total_predictions,
                'individual_predictions': individual_count,
                'batch_predictions': batch_count,
                'users_with_predictions': unique_users
            })
        
        return jsonify({
            'status': 'success',
            'system_info': info
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in get_system_info endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500


@prediction_bp.route('/history/<int:history_id>', methods=['DELETE'])
@login_required
def delete_prediction_history(history_id):
    """
    Delete a specific prediction history record.
    
    Users can only delete their own records unless they are admin.
    """
    try:
        # Find the history record
        history_record = PredictionHistory.query.get(history_id)
        
        if not history_record:
            return jsonify({
                'error': 'Prediction history record not found',
                'status': 'error'
            }), 404
        
        # Check permissions
        if history_record.user_id != current_user.id and not current_user.is_admin:
            return jsonify({
                'error': 'Insufficient permissions to delete this record',
                'status': 'error'
            }), 403
        
        # Delete the record
        db.session.delete(history_record)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Prediction history record deleted',
            'deleted_id': history_id
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in delete_prediction_history endpoint: {e}")
        db.session.rollback()
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500


# Error handlers for the blueprint
@prediction_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404


@prediction_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'status': 'error'
    }), 405


# Initialize the prediction system when blueprint is imported
def init_prediction_system(app):
    """Initialize the prediction system with app context."""
    try:
        # We're already in app context when this is called
        advanced_prediction_system.initialize(app.instance_path)
        app.logger.info("Advanced prediction system initialized successfully")
    except Exception as e:
        app.logger.warning(f"Prediction system initialization warning: {e}")
        # Continue anyway - system will work in demo mode 