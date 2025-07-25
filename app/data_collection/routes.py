"""
Data Collection Routes

Handles the API endpoints for survey data collection.
"""

from flask import request, jsonify, render_template
from flask_login import current_user
from . import bp
from .validators import SurveyValidator, ResponseValidator
from .models import DataStorageManager, SurveyData
from .exporters import ExportManager


@bp.route('/')
@bp.route('/dashboard')
def dashboard():
    """Data collection dashboard page."""
    return render_template('data_collection/dashboard.html')

@bp.route('/validation')
def data_validation():
    """Data validation testing page."""
    return render_template('data_collection/validation.html')

@bp.route('/export')
def export_center():
    """Data export center page."""
    return render_template('data_collection/export.html')

@bp.route('/history')
def export_history():
    """Export history page."""
    return render_template('data_collection/history.html')



@bp.route('/surveys', methods=['GET'])
def get_surveys():
    """Retrieve list of available surveys."""
    from app.models import Survey
    surveys = Survey.query.filter_by(is_active=True).all()
    
    survey_list = []
    for survey in surveys:
        survey_list.append({
            'id': survey.id,
            'title': survey.title,
            'description': survey.description,
            'questions': survey.get_questions(),
            'created_at': survey.created_at.isoformat() if survey.created_at else None
        })
    
    return jsonify({
        'surveys': survey_list,
        'message': 'Data collection system active'
    })

@bp.route('/surveys/submit', methods=['POST'])
def submit_survey():
    """Submit survey response with data storage."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'valid': False
            }), 400
        
        survey_id = data.get('survey_id')
        if not survey_id:
            return jsonify({
                'error': 'Survey ID is required',
                'valid': False
            }), 400
        
        
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
        
        if not validation_result['valid']:
            return jsonify({
                'message': 'Validation failed',
                'valid': False,
                'errors': validation_result['errors']
            }), 400
        
        
        metadata = {
            'ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent'),
            'timestamp': data.get('timestamp'),
            'validation_passed': True
        }
        
        
        user_id = current_user.id if current_user.is_authenticated else None
        storage_id = DataStorageManager.store_survey_submission(
            survey_id=survey_id,
            response_data=data,
            user_id=user_id,
            metadata=metadata
        )
        
        
        from app.data_collection.models import SurveyData
        survey_data = SurveyData.query.get(storage_id)
        if survey_data:
            survey_data.mark_as_processed(data)  
            from app import db
            db.session.commit()
        
        return jsonify({
            'message': 'Survey submitted and processed successfully',
            'valid': True,
            'storage_id': storage_id,
            'status': 'processed'
        })
        
    except Exception as e:
        
        try:
            if 'storage_id' in locals():
                from app.data_collection.models import SurveyData
                survey_data = SurveyData.query.get(storage_id)
                if survey_data:
                    survey_data.mark_as_failed(str(e))
                    from app import db
                    db.session.commit()
        except:
            pass  
            
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'valid': False
        }), 500

@bp.route('/surveys/validate', methods=['POST'])
def validate_survey_data():
    """Endpoint to validate survey data without submitting."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'valid': False
            }), 400
        
        
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

@bp.route('/surveys/<int:survey_id>/statistics', methods=['GET'])
def get_survey_statistics(survey_id):
    """Get statistics for a specific survey."""
    try:
        stats = DataStorageManager.get_response_statistics(survey_id)
        return jsonify({
            'survey_id': survey_id,
            'statistics': stats,
            'message': 'Statistics retrieved successfully'
        })
    except Exception as e:
        return jsonify({
            'error': f'Error retrieving statistics: {str(e)}'
        }), 500

@bp.route('/surveys/<int:survey_id>/responses', methods=['GET'])
def get_survey_responses(survey_id):
    """Get responses for a specific survey."""
    try:
        
        limit = request.args.get('limit', 100, type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        responses = DataStorageManager.get_survey_data(
            survey_id=survey_id,
            start_date=start_dt,
            end_date=end_dt,
            limit=limit
        )
        
        response_data = []
        for response in responses:
            response_data.append({
                'id': response.id,
                'submission_time': response.submission_time.isoformat(),
                'processing_status': response.processing_status,
                'data': response.get_raw_data(),
                'metadata': response.get_metadata()
            })
        
        return jsonify({
            'survey_id': survey_id,
            'responses': response_data,
            'count': len(response_data),
            'message': 'Responses retrieved successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error retrieving responses: {str(e)}'
        }), 500

@bp.route('/surveys/<int:survey_id>/responses/view')
def view_survey_responses(survey_id):
    """HTML page for viewing survey responses."""
    return render_template('data_collection/responses.html', survey_id=survey_id)

@bp.route('/surveys/<int:survey_id>/export/<export_format>', methods=['GET'])
def export_survey_data(survey_id, export_format):
    """Export survey data in specified format (csv, json, excel)."""
    try:
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        
        exported_by = current_user.id if current_user.is_authenticated else None
        
        
        export_result = ExportManager.create_export(
            survey_id=survey_id,
            export_format=export_format,
            start_date=start_dt,
            end_date=end_dt,
            exported_by=exported_by
        )
        
        if export_result['success']:
            return ExportManager.create_flask_response(export_result)
        else:
            return jsonify({
                'error': export_result['error'],
                'message': 'Export failed'
            }), 500
            
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'message': 'Invalid export format. Supported formats: csv, json, excel'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Export error: {str(e)}'
        }), 500

@bp.route('/exports/history', methods=['GET'])
def get_export_history():
    """Get history of data exports."""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = ExportManager.get_export_history(limit=limit)
        
        return jsonify({
            'export_history': history,
            'count': len(history),
            'message': 'Export history retrieved successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error retrieving export history: {str(e)}'
        }), 500

@bp.route('/surveys/<int:survey_id>/export/preview', methods=['GET'])
def preview_export_data(survey_id):
    """Preview data that would be exported for a survey."""
    try:
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 10, type=int)  
        
        
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        responses = DataStorageManager.get_survey_data(
            survey_id=survey_id,
            start_date=start_dt,
            end_date=end_dt,
            limit=limit
        )
        
        
        total_responses = DataStorageManager.get_survey_data(
            survey_id=survey_id,
            start_date=start_dt,
            end_date=end_dt
        )
        
        preview_data = []
        for response in responses:
            preview_data.append({
                'id': response.id,
                'submission_time': response.submission_time.isoformat(),
                'processing_status': response.processing_status,
                'raw_data': response.get_raw_data(),
                'metadata': response.get_metadata()
            })
        
        return jsonify({
            'survey_id': survey_id,
            'preview_data': preview_data,
            'preview_count': len(preview_data),
            'total_available': len(total_responses),
            'message': 'Export preview generated successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating preview: {str(e)}'
        }), 500 