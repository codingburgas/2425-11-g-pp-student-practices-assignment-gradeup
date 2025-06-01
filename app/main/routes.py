from flask import render_template, redirect, url_for, request, jsonify, flash, current_app
from flask_login import login_required, current_user
from app.main import bp
from app.models import School, Program, Survey
from app.preprocessing import PreprocessingPipeline
from app.preprocessing.utils import (
    load_survey_responses_to_dataframe, 
    validate_dataframe_for_preprocessing,
    export_processed_data,
    create_sample_survey_data
)
import os
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/dashboard')
@login_required
def dashboard():
    return render_template('main/dashboard.html')

@bp.route('/universities')
def universities():
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '', type=str)
    
    query = School.query
    if search:
        query = query.filter(School.name.contains(search) | School.location.contains(search))
    
    universities = query.order_by(School.name.asc()).paginate(
        page=page, per_page=12, error_out=False)
    
    return render_template('main/universities.html', 
                           title='Universities', 
                           universities=universities, 
                           search=search)

@bp.route('/survey')
def survey():
    surveys = Survey.query.filter_by(is_active=True).order_by(Survey.created_at.desc()).all()
    return render_template('main/survey_list.html', title='Available Surveys', surveys=surveys)

@bp.route('/specialties')
def specialties():
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '', type=str)
    university_id = request.args.get('university_id', type=int)
    
    query = Program.query.join(School)
    if search:
        query = query.filter(Program.name.contains(search) | 
                           Program.description.contains(search) |
                           Program.degree_type.contains(search))
    if university_id:
        query = query.filter(Program.school_id == university_id)
    
    programs = query.order_by(Program.name.asc()).paginate(
        page=page, per_page=12, error_out=False)
    
    universities = School.query.order_by(School.name.asc()).all()
    
    return render_template('main/specialties.html',
                           title='Academic Programs', 
                           programs=programs, 
                           universities=universities,
                           search=search,
                           selected_university=university_id)

@bp.route('/recommendations')
@login_required
def recommendations():
    return render_template('main/recommendations.html')

@bp.route('/favorites')
@login_required
def favorites():
    return render_template('main/favorites.html')

@bp.route('/survey/start')
@login_required
def start_survey():
    # Redirect to data collection survey list
    return redirect(url_for('data_collection.survey_list'))

@bp.route('/survey/take/<int:survey_id>')
@login_required
def take_survey(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    if not survey.is_active:
        flash('This survey is no longer available.', 'warning')
        return redirect(url_for('main.survey'))
    
    return render_template('main/take_survey.html', 
                         title=f'Take Survey: {survey.title}', 
                         survey=survey)

@bp.route('/survey/submit/<int:survey_id>', methods=['POST'])
@login_required
def submit_survey_response(survey_id):
    from app.models import SurveyResponse
    from app import db
    import json
    from flask import flash, request
    
    survey = Survey.query.get_or_404(survey_id)
    
    # Check if user already submitted this survey
    existing_response = SurveyResponse.query.filter_by(
        user_id=current_user.id, 
        survey_id=survey_id
    ).first()
    
    if existing_response:
        flash('You have already submitted this survey.', 'info')
        return redirect(url_for('main.survey'))
    
    # Collect answers from form
    answers = {}
    questions = survey.get_questions()
    
    for question in questions:
        q_id = str(question['id'])
        if question['type'] == 'multiple_choice':
            answers[q_id] = request.form.get(f'question_{q_id}')
        elif question['type'] == 'multiple_select':
            answers[q_id] = request.form.getlist(f'question_{q_id}')
        elif question['type'] in ['rating', 'slider']:
            answers[q_id] = int(request.form.get(f'question_{q_id}', 0))
        else:
            answers[q_id] = request.form.get(f'question_{q_id}')
    
    # Create and save response
    response = SurveyResponse(
        user_id=current_user.id,
        survey_id=survey_id,
        answers=json.dumps(answers)
    )
    
    try:
        db.session.add(response)
        db.session.commit()
        flash('Thank you! Your survey response has been submitted successfully.', 'success')
        return redirect(url_for('main.recommendations'))
    except Exception as e:
        db.session.rollback()
        flash('An error occurred while submitting your response. Please try again.', 'danger')
        return redirect(url_for('main.take_survey', survey_id=survey_id))

@bp.route('/data-preprocessing')
@login_required
def data_preprocessing():
    """Data preprocessing dashboard."""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('main.index'))
    
    # Get available surveys
    surveys = Survey.query.filter_by(is_active=True).all()
    
    # Get preprocessing statistics
    try:
        # Load a sample of survey data for preview
        sample_df = load_survey_responses_to_dataframe()
        
        stats = {
            'total_responses': len(sample_df) if not sample_df.empty else 0,
            'total_columns': len(sample_df.columns) if not sample_df.empty else 0,
            'has_data': not sample_df.empty
        }
        
        if not sample_df.empty:
            validation_report = validate_dataframe_for_preprocessing(sample_df)
            stats['data_quality'] = validation_report
        else:
            stats['data_quality'] = None
            
    except Exception as e:
        logger.error(f"Error loading preprocessing stats: {e}")
        stats = {'has_data': False, 'error': str(e)}
    
    return render_template('admin/data_preprocessing.html', 
                         surveys=surveys, 
                         stats=stats)

@bp.route('/api/preprocess-data', methods=['POST'])
@login_required
def api_preprocess_data():
    """API endpoint to run data preprocessing pipeline."""
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # Get parameters from request
        data = request.get_json()
        survey_id = data.get('survey_id')
        preprocessing_config = data.get('config', {})
        
        # Load survey data
        if survey_id:
            df = load_survey_responses_to_dataframe(survey_id=survey_id)
        else:
            df = load_survey_responses_to_dataframe()
        
        if df.empty:
            return jsonify({'error': 'No survey data found'}), 400
        
        # Initialize preprocessing pipeline
        pipeline = PreprocessingPipeline(
            missing_threshold=preprocessing_config.get('missing_threshold', 0.3),
            outlier_method=preprocessing_config.get('outlier_method', 'iqr'),
            numerical_method=preprocessing_config.get('numerical_method', 'standard'),
            categorical_method=preprocessing_config.get('categorical_method', 'onehot'),
            create_interactions=preprocessing_config.get('create_interactions', True),
            create_polynomials=preprocessing_config.get('create_polynomials', False),
            create_aggregations=preprocessing_config.get('create_aggregations', True),
            create_domain_features=preprocessing_config.get('create_domain_features', True),
            save_artifacts=True,
            artifacts_dir=os.path.join(current_app.root_path, '..', 'preprocessing_artifacts')
        )
        
        # Run preprocessing
        processed_df = pipeline.fit_transform(df, target_column='recommendation_score')
        
        # Get pipeline report
        report = pipeline.get_pipeline_report()
        
        # Export processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"processed_survey_data_{timestamp}.csv"
        export_path = os.path.join(current_app.root_path, '..', 'exports', export_filename)
        
        # Create exports directory if it doesn't exist
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        export_processed_data(processed_df, export_path, format='csv')
        
        return jsonify({
            'success': True,
            'report': {
                'original_shape': report['original_shape'],
                'final_shape': report['final_shape'],
                'processing_time': report.get('processing_time_seconds', 0),
                'features_created': len(report.get('feature_engineering', {}).get('created_features', [])),
                'data_quality': report.get('validation', {}).get('is_valid', False)
            },
            'export_filename': export_filename,
            'summary': pipeline.get_summary()
        })
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/preprocessing-sample', methods=['POST'])
@login_required
def api_create_preprocessing_sample():
    """API endpoint to create sample data for testing preprocessing."""
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        data = request.get_json()
        num_responses = data.get('num_responses', 100)
        num_questions = data.get('num_questions', 10)
        
        # Create sample data
        sample_df = create_sample_survey_data(num_responses, num_questions)
        
        # Export sample data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_filename = f"sample_survey_data_{timestamp}.csv"
        sample_path = os.path.join(current_app.root_path, '..', 'exports', sample_filename)
        
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        export_processed_data(sample_df, sample_path, format='csv')
        
        # Get validation report
        validation_report = validate_dataframe_for_preprocessing(sample_df)
        
        return jsonify({
            'success': True,
            'filename': sample_filename,
            'shape': sample_df.shape,
            'columns': list(sample_df.columns),
            'validation': validation_report
        })
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/preprocessing-status/<survey_id>')
@login_required
def api_preprocessing_status(survey_id):
    """API endpoint to get preprocessing status for a survey."""
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # Load survey data
        df = load_survey_responses_to_dataframe(survey_id=int(survey_id))
        
        if df.empty:
            return jsonify({
                'has_data': False,
                'message': 'No survey responses found'
            })
        
        # Get validation report
        validation_report = validate_dataframe_for_preprocessing(df)
        
        # Calculate basic statistics
        stats = {
            'has_data': True,
            'shape': df.shape,
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'validation': validation_report
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting preprocessing status: {e}")
        return jsonify({'error': str(e)}), 500 