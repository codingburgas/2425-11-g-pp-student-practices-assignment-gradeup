from flask import render_template, redirect, url_for, request, jsonify, flash, current_app, abort
from flask_login import login_required, current_user
from flask_wtf.csrf import generate_csrf
from sqlalchemy import or_, and_, desc, asc, func
from app.main import bp
from app.main.forms import UserSearchForm
from app.models import School, Program, Survey, SurveyResponse, Favorite, Recommendation, User
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
from app.ml.recommendation_engine import recommendation_engine

logger = logging.getLogger(__name__)

@bp.route('/')
@bp.route('/index')
def index():
    return render_template('index.html', title='Home')

@bp.route('/status')
def simple_status():
    """Simple status check that always works"""
    return jsonify({
        "status": "app_running",
        "message": "Flask app is running successfully",
        "timestamp": datetime.now().isoformat()
    })

@bp.route('/health')
def health_check():
    """
    Health check endpoint for debugging
    
    Fixed for SQL Server compatibility:
    - Combined queries to avoid connection busy issues
    - Added fallback logic for database checks
    """
    try:
        # Check if DATABASE_URL is set
        db_url = os.environ.get('DATABASE_URL')
        db_status = "✅ Set" if db_url else "❌ Not set"
        
        # Test database connection with simple approach
        from sqlalchemy import text
        
        db_connection = "❌ Failed"
        table_count = 0
        
        try:
            # Use a single combined query to avoid connection busy issues
            result = db.session.execute(text("""
                SELECT 
                    1 as connection_test,
                    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
                     WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG = DB_NAME()) as table_count
            """))
            row = result.fetchone()
            
            if row and row[0] == 1:
                db_connection = "✅ Connected"
                table_count = row[1] if row[1] is not None else 0
            
        except Exception as conn_error:
            # If SQL Server query fails, try a simpler test
            try:
                simple_result = db.session.execute(text('SELECT 1'))
                if simple_result.scalar() == 1:
                    db_connection = "✅ Connected (basic)"
                    # Try to count User table as fallback
                    user_count = User.query.count()
                    table_count = f"Users: {user_count}"
            except Exception:
                db_connection = f"❌ Failed: {str(conn_error)[:100]}"
        
        return jsonify({
            "status": "healthy",
            "database_url": db_status,
            "database_connection": db_connection,
            "table_count": table_count,
            "environment": "azure" if db_url else "local"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "database_url": db_status if 'db_status' in locals() else "unknown",
            "error": str(e)[:200],  # Limit error message length
            "environment": "azure" if os.environ.get('DATABASE_URL') else "local"
        }), 500

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
    
    # Get personalized suggestions from recommendation engine
    personalized_suggestions = []
    try:
        suggestions = recommendation_engine.get_personalized_suggestions(
            user_id=current_user.id,
            limit=3
        )
        # Convert completion suggestions to dashboard format
        for suggestion in suggestions.get('completion_suggestions', []):
            personalized_suggestions.append({
                'icon': 'fas fa-' + ('poll' if suggestion['type'] == 'survey' else 
                                   'star' if suggestion['type'] == 'favorites' else 'user-edit'),
                'title': suggestion['title'],
                'description': suggestion['description'],
                'link': suggestion['action_url'],
                'priority': suggestion['priority']
            })
    except Exception as e:
        logger.warning(f"Error getting personalized suggestions: {e}")
    
    # Next steps suggestions (fallback if recommendation engine fails)
    next_steps = personalized_suggestions if personalized_suggestions else []
    
    if not next_steps:
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

    # Get quick recommendations for dashboard
    quick_recommendations = []
    try:
        user_responses = SurveyResponse.query.filter_by(user_id=current_user.id).all()
        if user_responses:
            latest_response = user_responses[-1]
            raw_survey_data = latest_response.get_answers()
            
            # Map survey data to format expected by recommendation engine
            from app.ml.utils import map_survey_data_to_recommendation_format
            survey_data = map_survey_data_to_recommendation_format(raw_survey_data)
            
            quick_recs = recommendation_engine.recommend_programs(
                user_id=current_user.id,
                survey_data=survey_data,
                user_preferences=current_user.get_preferences() if current_user.preferences else None,
                top_k=3
            )
            quick_recommendations = quick_recs
    except Exception as e:
        logger.warning(f"Error getting quick recommendations: {e}")
    
    dashboard_data = {
        'total_surveys': total_surveys,
        'completed_surveys': completed_surveys,
        'survey_progress': survey_progress,
        'profile_completion': profile_completion,
        'overall_progress': overall_progress,
        'favorites_count': favorites_count,
        'recent_recommendations': recent_recommendations,
        'recent_favorites': recent_favorites,
        'next_steps': next_steps[:3],  # Show top 3 suggestions
        'quick_recommendations': quick_recommendations
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

@bp.route('/users')
@login_required
def users():
    """User directory page with search and filtering"""
    page = request.args.get('page', 1, type=int)
    form = UserSearchForm()
    
    # Build base query - only show verified users
    query = User.query.filter_by(email_verified=True)
    
    # Apply search filter
    search = request.args.get('search', '', type=str)
    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            or_(
                User.username.ilike(search_filter),
                User.location.ilike(search_filter),
                User.bio.ilike(search_filter)
            )
        )
    
    # Apply location filter
    location_filter = request.args.get('location_filter', '', type=str)
    if location_filter:
        query = query.filter(User.location == location_filter)
    
    # Apply sorting
    sort_by = request.args.get('sort_by', 'username', type=str)
    if sort_by == 'created_at_desc':
        query = query.order_by(User.created_at.desc())
    elif sort_by == 'created_at_asc':
        query = query.order_by(User.created_at.asc())
    elif sort_by == 'location':
        query = query.order_by(User.location.asc().nullslast(), User.username.asc())
    else:  # default to username
        query = query.order_by(User.username.asc())
    
    # Paginate results
    users_pagination = query.paginate(
        page=page, per_page=20, error_out=False
    )
    
    # Get statistics
    total_users = User.query.filter_by(email_verified=True).count()
    recent_users = User.query.filter_by(email_verified=True)\
        .order_by(User.created_at.desc()).limit(5).all()
    
    # Pre-populate form with current values
    form.search.data = search
    form.location_filter.data = location_filter
    form.sort_by.data = sort_by
    
    return render_template('main/users.html',
                         title='User Directory',
                         users=users_pagination,
                         form=form,
                         search=search,
                         location_filter=location_filter,
                         sort_by=sort_by,
                         total_users=total_users,
                         recent_users=recent_users)

@bp.route('/users/<username>')
@login_required
def user_profile(username):
    """View individual user profile"""
    # Case-insensitive username lookup
    user = User.query.filter(func.lower(User.username) == func.lower(username)).first()
    
    if not user:
        flash(f'User "{username}" not found.', 'error')
        return redirect(url_for('main.users'))
    
    # Check if user is verified and profile can be viewed
    if not user.email_verified:
        flash('This user profile is not available.', 'warning')
        return redirect(url_for('main.users'))
    
    if not user.can_view_profile(current_user):
        flash('You do not have permission to view this profile.', 'error')
        return redirect(url_for('main.users'))
    
    # Get user's public profile data
    profile_data = user.get_public_profile()
    
    # Get recent survey responses (count only for privacy)
    recent_survey_count = user.survey_responses.count()
    
    # Get favorite schools count
    favorites_count = user.favorites.count()
    
    # Check if viewing own profile
    is_own_profile = current_user.id == user.id
    
    # Get some related users (from same location or similar activity)
    related_users = []
    if user.location:
        related_users = User.query.filter(
            and_(
                User.location == user.location,
                User.id != user.id,
                User.email_verified == True
            )
        ).limit(3).all()
    
    return render_template('main/user_profile.html',
                         title=f'{user.get_display_name()}\' Profile',
                         user=user,
                         profile_data=profile_data,
                         recent_survey_count=recent_survey_count,
                         favorites_count=favorites_count,
                         is_own_profile=is_own_profile,
                         related_users=related_users)

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
    """Enhanced recommendations page with new recommendation engine."""
    try:
        # Get user's survey responses
        user_responses = SurveyResponse.query.filter_by(user_id=current_user.id).all()
        
        recommendations_data = {
            'program_recommendations': [],
            'university_recommendations': [],
            'personalized_suggestions': {},
            'recommendation_history': [],
            'user_responses': user_responses,
            'max_submissions': 3
        }
        
        # Get program recommendations if user has survey data
        if user_responses:
            latest_response = user_responses[-1]  # Get most recent response
            raw_survey_data = latest_response.get_answers()
            
            # Map survey data to format expected by recommendation engine
            from app.ml.utils import map_survey_data_to_recommendation_format
            survey_data = map_survey_data_to_recommendation_format(raw_survey_data)
            
            logger.info(f"Raw survey data: {raw_survey_data}")
            logger.info(f"Mapped survey data: {survey_data}")
            
            # Get program recommendations
            program_recs = recommendation_engine.recommend_programs(
                user_id=current_user.id,
                survey_data=survey_data,
                user_preferences=current_user.get_preferences() if current_user.preferences else None,
                top_k=8
            )
            recommendations_data['program_recommendations'] = program_recs
            
            # Get university recommendations
            university_recs = recommendation_engine.match_universities(
                user_preferences=current_user.get_preferences() if current_user.preferences else {},
                survey_data=survey_data,
                top_k=6
            )
            recommendations_data['university_recommendations'] = university_recs
            
            # Store recommendations in history
            if program_recs:
                recommendation_engine.store_recommendation_history(
                    user_id=current_user.id,
                    survey_response_id=latest_response.id,
                    recommendations=program_recs,
                    recommendation_type='program'
                )
        
        # Get personalized suggestions
        personalized = recommendation_engine.get_personalized_suggestions(
            user_id=current_user.id,
            limit=5
        )
        recommendations_data['personalized_suggestions'] = personalized
        
        # Get recommendation history
        history = recommendation_engine.get_recommendation_history(
            user_id=current_user.id,
            limit=10
        )
        recommendations_data['recommendation_history'] = history
        
        logger.info(f"Enhanced recommendations page for user {current_user.id}: "
                   f"Programs: {len(recommendations_data['program_recommendations'])}, "
                   f"Universities: {len(recommendations_data['university_recommendations'])}")
        
        return render_template('main/recommendations.html', **recommendations_data)
        
    except Exception as e:
        logger.error(f"Error in recommendations page: {e}")
        flash('Error loading recommendations. Please try again.', 'error')
        return render_template('main/recommendations.html', 
                             program_recommendations=[],
                             university_recommendations=[],
                             personalized_suggestions={},
                             recommendation_history=[],
                             user_responses=[],
                             max_submissions=3)

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

@bp.route('/ml-training')
@login_required
def ml_training():
    """ML Training interface for testing the training system"""
    return render_template('main/ml_training.html', title='ML Training System')

@bp.route('/api/ml-train', methods=['POST'])
@login_required
def api_ml_train():
    """API endpoint to run ML training with sample data"""
    try:
        from app.ml.training import LinearRegression, LogisticRegression, TrainingPipeline
        import numpy as np
        
        # Get parameters from request
        data = request.get_json()
        model_type = data.get('model_type', 'linear_regression')
        n_samples = data.get('n_samples', 1000)
        learning_rate = data.get('learning_rate', 0.01)
        max_iterations = data.get('max_iterations', 100)
        
        # Generate sample data
        np.random.seed(42)
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        
        if model_type == 'linear_regression':
            # Regression target
            y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1
            model_class = LinearRegression
        else:
            # Classification target
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
            model_class = LogisticRegression
        
        # Setup training pipeline
        pipeline = TrainingPipeline(experiment_name="web_demo", auto_tracking=False)
        
        # Prepare data
        pipeline.prepare_data(
            X, y, 
            test_size=0.2, 
            preprocessing_config={'scaling_method': 'standard'}
        )
        
        # Train model
        model_config = {
            'demo_model': {
                'class': model_class,
                'params': {
                    'learning_rate': learning_rate,
                    'max_iterations': max_iterations
                }
            }
        }
        
        results = pipeline.compare_models(model_config)
        
        # Get test metrics
        demo_model_info = pipeline.trained_models['demo_model']
        test_metrics = demo_model_info['test_metrics']
        
        return jsonify({
            'status': 'success',
            'model_type': model_type,
            'test_metrics': test_metrics,
            'training_time': demo_model_info['training_result'].training_time,
            'data_shape': [n_samples, n_features],
            'message': f'Successfully trained {model_type} model!'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@bp.route('/api/university-recommendation-demo', methods=['POST'])
@login_required
def api_university_recommendation_demo():
    """Demo API for university recommendation using ML"""
    try:
        from app.ml.training import LogisticRegression, TrainingPipeline
        import numpy as np
        
        # Get student preferences from request
        data = request.get_json()
        student_preferences = data.get('preferences', {})
        
        # Create mock training data (in real app, this would come from database)
        np.random.seed(42)
        n_students = 500
        n_features = 6  # Academic focus, location pref, budget, social, career, research
        
        # Generate training data
        X_training = np.random.rand(n_students, n_features)
        
        # Generate satisfaction based on weighted preferences
        satisfaction_scores = (
            X_training[:, 0] * 0.25 +  # Academic focus
            X_training[:, 1] * 0.20 +  # Location preference
            X_training[:, 2] * 0.15 +  # Budget fit
            X_training[:, 3] * 0.15 +  # Social environment
            X_training[:, 4] * 0.15 +  # Career prospects
            X_training[:, 5] * 0.10    # Research opportunities
        )
        y_satisfaction = (satisfaction_scores > 0.5).astype(int)
        
        # Train recommendation model
        pipeline = TrainingPipeline(experiment_name="uni_rec_demo", auto_tracking=False)
        pipeline.prepare_data(X_training, y_satisfaction, test_size=0.2)
        
        model_config = {
            'recommendation_model': {
                'class': LogisticRegression,
                'params': {'learning_rate': 0.1, 'max_iterations': 100}
            }
        }
        
        pipeline.compare_models(model_config)
        
        # Convert student preferences to feature vector
        student_vector = np.array([[
            student_preferences.get('academic_focus', 0.5),
            student_preferences.get('location_preference', 0.5), 
            student_preferences.get('budget_fit', 0.5),
            student_preferences.get('social_environment', 0.5),
            student_preferences.get('career_prospects', 0.5),
            student_preferences.get('research_opportunities', 0.5)
        ]])
        
        # Get prediction
        best_model_name, best_trainer = pipeline.get_best_model('accuracy')
        satisfaction_probability = best_trainer.predict(student_vector)[0]
        
        # Generate mock university recommendations
        universities = [
            {"name": "Sofia University", "match_score": satisfaction_probability * 0.95},
            {"name": "Technical University Sofia", "match_score": satisfaction_probability * 0.88},
            {"name": "UNWE", "match_score": satisfaction_probability * 0.82},
            {"name": "University of Plovdiv", "match_score": satisfaction_probability * 0.78},
            {"name": "Burgas Free University", "match_score": satisfaction_probability * 0.71}
        ]
        
        # Sort by match score
        universities.sort(key=lambda x: x['match_score'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'satisfaction_probability': float(satisfaction_probability),
            'recommendations': universities[:3],  # Top 3
            'model_accuracy': pipeline.trained_models['recommendation_model']['test_metrics']['accuracy'],
            'message': 'University recommendations generated successfully!'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 