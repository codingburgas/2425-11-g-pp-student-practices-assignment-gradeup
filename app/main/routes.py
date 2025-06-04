from flask import render_template, redirect, url_for, request, jsonify, flash, current_app
from flask_login import login_required, current_user
from flask_wtf.csrf import generate_csrf
from app.main import bp
from app.models import School, Program, Survey, SurveyResponse, Favorite, Recommendation
from app import db
# from app.preprocessing import PreprocessingPipeline
# from app.preprocessing.utils import (
#     load_survey_responses_to_dataframe, 
#     validate_dataframe_for_preprocessing,
#     export_processed_data,
#     create_sample_survey_data
# )
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
    # Calculate user progress metrics
    total_surveys = Survey.query.filter_by(is_active=True).count()
    completed_surveys = SurveyResponse.query.filter_by(user_id=current_user.id).count()
    
    # Profile completion
    profile_fields = [
        current_user.bio,
        current_user.location,
        current_user.preferences
    ]
    profile_completion = sum(1 for field in profile_fields if field) * 25  # Each field worth 25%
    
    # Recent recommendations
    recent_recommendations = db.session.query(Recommendation)\
        .join(SurveyResponse)\
        .filter(SurveyResponse.user_id == current_user.id)\
        .order_by(Recommendation.created_at.desc())\
        .limit(5).all()
    
    # Favorite schools count
    favorites_count = Favorite.query.filter_by(user_id=current_user.id).count()
    
    # Recent favorites
    recent_favorites = Favorite.query.filter_by(user_id=current_user.id)\
        .order_by(Favorite.created_at.desc())\
        .limit(3).all()
    
    # Calculate overall progress
    survey_progress = (completed_surveys / max(total_surveys, 1)) * 100
    overall_progress = (survey_progress + profile_completion) / 2
    
    # Next steps suggestions
    next_steps = []
    if completed_surveys == 0:
        next_steps.append({
            'icon': 'fas fa-poll',
            'title': 'Take Your First Survey',
            'description': 'Complete our survey to get personalized university recommendations',
            'link': url_for('main.survey'),
            'priority': 1
        })
    
    if not current_user.bio:
        next_steps.append({
            'icon': 'fas fa-user-edit',
            'title': 'Complete Your Profile',
            'description': 'Add a bio and preferences to improve recommendations',
            'link': url_for('auth.profile'),
            'priority': 2
        })
    
    if favorites_count == 0:
        next_steps.append({
            'icon': 'fas fa-star',
            'title': 'Explore Universities',
            'description': 'Browse universities and save your favorites',
            'link': url_for('main.universities'),
            'priority': 3
        })
    
    # Sort by priority
    next_steps.sort(key=lambda x: x['priority'])
    
    dashboard_data = {
        'total_surveys': total_surveys,
        'completed_surveys': completed_surveys,
        'survey_progress': survey_progress,
        'profile_completion': profile_completion,
        'overall_progress': overall_progress,
        'favorites_count': favorites_count,
        'recent_recommendations': recent_recommendations,
        'recent_favorites': recent_favorites,
        'next_steps': next_steps[:3]  # Show top 3 suggestions
    }
    
    return render_template('main/dashboard.html', **dashboard_data)

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
    # Check if there are any active surveys, if not create a sample one
    surveys = Survey.query.filter_by(is_active=True).order_by(Survey.created_at.desc()).all()
    
    # Create sample survey if none exist
    if not surveys:
        create_sample_survey()
        surveys = Survey.query.filter_by(is_active=True).order_by(Survey.created_at.desc()).all()
    
    return render_template('main/survey_list.html', title='Available Surveys', surveys=surveys)

@bp.route('/create-sample-survey')
@login_required 
def create_sample_survey():
    """Create a sample survey for testing"""
    import json
    
    # Check if survey already exists
    existing_survey = Survey.query.filter_by(title="Educational Preferences Survey").first()
    if existing_survey:
        flash('Sample survey already exists!', 'info')
        return redirect(url_for('main.survey'))
    
    survey_questions = [
        {
            "id": 1,
            "text": "What subjects do you enjoy the most?",
            "type": "multiple_choice",
            "required": True,
            "options": ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science", "History", "Literature", "Languages", "Arts", "Economics", "Psychology", "Philosophy"]
        },
        {
            "id": 2,
            "text": "What type of career are you interested in?",
            "type": "multiple_choice",
            "required": True,
            "options": ["Technology", "Science", "Medicine", "Business", "Law", "Education", "Arts", "Engineering", "Social Services", "Government"]
        },
        {
            "id": 3,
            "text": "How important is university location to you?",
            "type": "rating",
            "required": True,
            "min": 1,
            "max": 5,
            "labels": {"1": "Not Important", "5": "Very Important"}
        },
        {
            "id": 4,
            "text": "Do you prefer theoretical or practical learning?",
            "type": "slider",
            "required": True,
            "min": 1,
            "max": 5,
            "labels": ["Theoretical", "Balanced", "Practical"]
        },
        {
            "id": 5,
            "text": "What program length do you prefer?",
            "type": "multiple_choice",
            "required": True,
            "options": ["3 years", "4 years", "5 years", "6 years"]
        },
        {
            "id": 6,
            "text": "How important is employment rate after graduation?",
            "type": "rating",
            "required": True,
            "min": 1,
            "max": 5,
            "labels": {"1": "Not Important", "5": "Very Important"}
        },
        {
            "id": 7,
            "text": "What teaching language do you prefer?",
            "type": "multiple_choice",
            "required": True,
            "options": ["Bulgarian", "English", "Other", "Doesn't matter"]
        },
        {
            "id": 8,
            "text": "What are your average grades in high school?",
            "type": "multiple_choice",
            "required": True,
            "options": ["Below 4.0", "4.0-4.5", "4.5-5.0", "5.0-5.5", "5.5-6.0"]
        },
        {
            "id": 9,
            "text": "What extracurricular activities do you enjoy?",
            "type": "multiple_select",
            "required": False,
            "options": ["Sports", "Music", "Art", "Programming", "Volunteering", "Student Government", "Debate", "None"]
        },
        {
            "id": 10,
            "text": "What skills would you like to develop?",
            "type": "multiple_select",
            "required": False,
            "options": ["Problem Solving", "Critical Thinking", "Communication", "Leadership", "Teamwork", "Technical Skills", "Creative Thinking", "Research Skills"]
        }
    ]
    
    survey = Survey(
        title="Educational Preferences Survey",
        description="This survey helps us understand your educational preferences and career goals to recommend suitable academic programs.",
        questions=json.dumps(survey_questions),
        is_active=True
    )
    
    try:
        db.session.add(survey)
        db.session.commit()
        flash('Sample survey created successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error creating survey: {str(e)}', 'danger')
    
    return redirect(url_for('main.survey'))

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
    from app.models import SurveyResponse
    
    # Calculate retakes information
    MAX_SUBMISSIONS = 3
    user_submissions = SurveyResponse.query.filter_by(user_id=current_user.id).count()
    retakes_left = max(0, MAX_SUBMISSIONS - user_submissions)
    has_retakes = retakes_left > 0
    
    # Explicitly get user's survey responses for the template
    user_responses = SurveyResponse.query.filter_by(user_id=current_user.id).order_by(SurveyResponse.created_at.desc()).all()
    
    logger.info(f"Recommendations page for user {current_user.id}: found {len(user_responses)} responses")
    
    return render_template('main/recommendations.html', 
                         user_submissions=user_submissions,
                         retakes_left=retakes_left,
                         has_retakes=has_retakes,
                         max_submissions=MAX_SUBMISSIONS,
                         user_responses=user_responses)

@bp.route('/favorites')
@login_required
def favorites():
    return render_template('main/favorites.html')

@bp.route('/survey/start')
@login_required
def start_survey():
    
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
                         survey=survey,
                         csrf_token=generate_csrf())

@bp.route('/survey/submit/<int:survey_id>', methods=['POST'])
@login_required
def submit_survey_response(survey_id):
    from app.models import SurveyResponse
    import json
    
    logger.info(f"Survey submission started for survey_id={survey_id}, user_id={current_user.id}")
    
    survey = Survey.query.get_or_404(survey_id)
    logger.info(f"Survey found: {survey.title}")
    
    # Check user's retake count (max 3 retakes allowed)
    existing_responses_count = SurveyResponse.query.filter_by(
        user_id=current_user.id, 
        survey_id=survey_id
    ).count()
    
    logger.info(f"Existing responses count: {existing_responses_count}")
    
    # Allow up to 3 total submissions (original + 2 retakes)
    MAX_SUBMISSIONS = 3
    
    if existing_responses_count >= MAX_SUBMISSIONS:
        logger.warning(f"User {current_user.id} exceeded max submissions for survey {survey_id}")
        flash(f'⚠️ You have reached the maximum number of survey submissions ({MAX_SUBMISSIONS}). Your latest responses are being used for recommendations.', 'warning')
        return redirect(url_for('main.recommendations'))
    
    # Process survey responses
    answers = {}
    questions = survey.get_questions()
    required_missing = []
    
    logger.info(f"Processing {len(questions)} questions")
    logger.info(f"Form data keys: {list(request.form.keys())}")
    
    for question in questions:
        q_id = str(question['id'])
        question_name = f'question_{q_id}'
        
        logger.info(f"Processing question {q_id}: {question.get('text', 'No text')[:50]}...")
        
        if question['type'] == 'multiple_choice':
            answer = request.form.get(question_name)
            logger.info(f"Multiple choice answer for {question_name}: {answer}")
            if answer:
                answers[q_id] = answer
            elif question.get('required', True):
                required_missing.append(question['text'])
                
        elif question['type'] == 'multiple_select':
            answer_list = request.form.getlist(question_name)
            logger.info(f"Multiple select answers for {question_name}: {answer_list}")
            if answer_list:
                answers[q_id] = answer_list
            elif question.get('required', True):
                required_missing.append(question['text'])
                
        elif question['type'] in ['rating', 'slider']:
            answer = request.form.get(question_name)
            logger.info(f"Rating/slider answer for {question_name}: {answer}")
            if answer:
                try:
                    answers[q_id] = int(answer)
                except ValueError:
                    answers[q_id] = 0
            elif question.get('required', True):
                required_missing.append(question['text'])
                
        else:  # text input
            answer = request.form.get(question_name, '').strip()
            logger.info(f"Text answer for {question_name}: {answer[:50] if answer else 'Empty'}...")
            if answer:
                answers[q_id] = answer
            elif question.get('required', True):
                required_missing.append(question['text'])
    
    logger.info(f"Processed answers: {answers}")
    logger.info(f"Required missing: {required_missing}")
    
    # Check for required questions
    if required_missing:
        logger.warning(f"Missing required answers: {required_missing}")
        flash(f'Please answer all required questions: {", ".join(required_missing)}', 'warning')
        return redirect(url_for('main.take_survey', survey_id=survey_id))
    
    # Create new survey response (allowing multiple responses)
    submission_number = existing_responses_count + 1
    response = SurveyResponse(
        user_id=current_user.id,
        survey_id=survey_id,
        answers=json.dumps(answers)
    )
    
    logger.info(f"Created SurveyResponse object: user_id={response.user_id}, survey_id={response.survey_id}")
    
    try:
        db.session.add(response)
        logger.info("Added response to session")
        
        db.session.commit()
        logger.info("Successfully committed response to database")
        
        # Verify the save by querying back
        saved_response = SurveyResponse.query.filter_by(
            user_id=current_user.id,
            survey_id=survey_id
        ).order_by(SurveyResponse.created_at.desc()).first()
        
        if saved_response:
            logger.info(f"Verified save: Response ID {saved_response.id} saved at {saved_response.created_at}")
        else:
            logger.error("Failed to verify saved response!")
        
        # Show different messages based on submission number
        if submission_number == 1:
            flash('🎉 Thank you! Your survey has been submitted successfully. You can now view your personalized recommendations!', 'success')
        else:
            retakes_left = MAX_SUBMISSIONS - submission_number
            if retakes_left > 0:
                flash(f'✨ Survey retake #{submission_number-1} completed! Your recommendations have been updated. You have {retakes_left} retake(s) remaining.', 'success')
            else:
                flash(f'🌟 Final survey submission completed! Your recommendations have been updated. You have used all {MAX_SUBMISSIONS} submissions.', 'info')
        
        # Try to generate recommendations immediately
        try:
            from app.ml.service import MLModelService
            ml_service = MLModelService()
            recommendations = ml_service.get_program_recommendations(current_user.id)
            if recommendations:
                flash(f'✨ Great news! We found {len(recommendations)} program recommendations based on your latest responses.', 'info')
        except Exception as ml_error:
            # Don't fail the whole process if ML fails
            logger.warning(f"ML recommendation generation failed: {ml_error}")
        
        return redirect(url_for('main.recommendations'))
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving survey response: {e}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception args: {e.args}")
        flash('An error occurred while submitting your response. Please try again.', 'danger')
        return redirect(url_for('main.take_survey', survey_id=survey_id))

@bp.route('/data-preprocessing')
@login_required
def data_preprocessing():
    """Data preprocessing dashboard."""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('main.index'))
    
    # Temporarily disabled - missing dependencies
    flash('Data preprocessing is temporarily disabled due to missing dependencies.', 'warning')
    return redirect(url_for('main.dashboard'))
    
    # surveys = Survey.query.filter_by(is_active=True).all()
    
    # try:
    #     sample_df = load_survey_responses_to_dataframe()
        
    #     stats = {
    #         'total_responses': len(sample_df) if not sample_df.empty else 0,
    #         'total_columns': len(sample_df.columns) if not sample_df.empty else 0,
    #         'has_data': not sample_df.empty
    #     }
        
    #     if not sample_df.empty:
    #         validation_report = validate_dataframe_for_preprocessing(sample_df)
    #         stats['data_quality'] = validation_report
    #     else:
    #         stats['data_quality'] = None
            
    # except Exception as e:
    #     logger.error(f"Error loading preprocessing stats: {e}")
    #     stats = {'has_data': False, 'error': str(e)}
    
    # return render_template('admin/data_preprocessing.html', 
    #                      surveys=surveys, 
    #                      stats=stats)

@bp.route('/api/preprocess-data', methods=['POST'])
@login_required
def api_preprocess_data():
    """API endpoint to run data preprocessing pipeline."""
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    return jsonify({'error': 'Data preprocessing temporarily disabled'}), 503
    
    # try:
    #     data = request.get_json()
    #     survey_id = data.get('survey_id')
    #     preprocessing_config = data.get('config', {})
        
    #     if survey_id:
    #         df = load_survey_responses_to_dataframe(survey_id=survey_id)
    #     else:
    #         df = load_survey_responses_to_dataframe()
        
    #     if df.empty:
    #         return jsonify({'error': 'No survey data found'}), 400
        
    #     pipeline = PreprocessingPipeline(
    #         missing_threshold=preprocessing_config.get('missing_threshold', 0.3),
    #         outlier_method=preprocessing_config.get('outlier_method', 'iqr'),
    #         numerical_method=preprocessing_config.get('numerical_method', 'standard'),
    #         categorical_method=preprocessing_config.get('categorical_method', 'onehot'),
    #         create_interactions=preprocessing_config.get('create_interactions', True),
    #         create_polynomials=preprocessing_config.get('create_polynomials', False),
    #         create_aggregations=preprocessing_config.get('create_aggregations', True),
    #         create_domain_features=preprocessing_config.get('create_domain_features', True),
    #         save_artifacts=True,
    #         artifacts_dir=os.path.join(current_app.root_path, '..', 'preprocessing_artifacts')
    #     )
        
    #     processed_df = pipeline.fit_transform(df, target_column='recommendation_score')
        
    #     report = pipeline.get_pipeline_report()
        
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     export_filename = f"processed_survey_data_{timestamp}.csv"
    #     export_path = os.path.join(current_app.root_path, '..', 'exports', export_filename)
        
    #     os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
    #     export_processed_data(processed_df, export_path, format='csv')
        
    #     return jsonify({
    #         'success': True,
    #         'report': {
    #             'original_shape': report['original_shape'],
    #             'final_shape': report['final_shape'],
    #             'processing_time': report.get('processing_time_seconds', 0),
    #             'features_created': len(report.get('feature_engineering', {}).get('created_features', [])),
    #             'data_quality': report.get('validation', {}).get('is_valid', False)
    #         },
    #         'export_filename': export_filename,
    #         'summary': pipeline.get_summary()
    #     })
        
    # except Exception as e:
    #     logger.error(f"Error in data preprocessing: {e}")
    #     return jsonify({'error': str(e)}), 500

@bp.route('/api/preprocessing-sample', methods=['POST'])
@login_required
def api_create_preprocessing_sample():
    """API endpoint to create sample data for testing preprocessing."""
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    return jsonify({'error': 'Data preprocessing temporarily disabled'}), 503
    
    # try:
    #     data = request.get_json()
    #     num_responses = data.get('num_responses', 100)
    #     num_questions = data.get('num_questions', 10)
        
    #     sample_df = create_sample_survey_data(num_responses, num_questions)
        
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     sample_filename = f"sample_survey_data_{timestamp}.csv"
    #     sample_path = os.path.join(current_app.root_path, '..', 'exports', sample_filename)
        
    #     os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    #     export_processed_data(sample_df, sample_path, format='csv')
        
    #     validation_report = validate_dataframe_for_preprocessing(sample_df)
        
    #     return jsonify({
    #         'success': True,
    #         'filename': sample_filename,
    #         'shape': sample_df.shape,
    #         'columns': list(sample_df.columns),
    #         'validation': validation_report
    #     })
        
    # except Exception as e:
    #     logger.error(f"Error creating sample data: {e}")
    #     return jsonify({'error': str(e)}), 500

@bp.route('/api/preprocessing-status/<survey_id>')
@login_required
def api_preprocessing_status(survey_id):
    """API endpoint to get preprocessing status for a survey."""
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    return jsonify({'error': 'Data preprocessing temporarily disabled'}), 503
    
    # try:
    #     df = load_survey_responses_to_dataframe(survey_id=int(survey_id))
        
    #     if df.empty:
    #         return jsonify({
    #             'has_data': False,
    #             'message': 'No survey responses found'
    #         })
        
    #     validation_report = validate_dataframe_for_preprocessing(df)
        
    #     stats = {
    #         'has_data': True,
    #         'shape': df.shape,
    #         'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
    #         'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
    #         'categorical_columns': len(df.select_dtypes(include=['object']).columns),
    #         'validation': validation_report
    #     }
        
    #     return jsonify(stats)
        
    # except Exception as e:
    #     logger.error(f"Error getting preprocessing status: {e}")
    #     return jsonify({'error': str(e)}), 500 