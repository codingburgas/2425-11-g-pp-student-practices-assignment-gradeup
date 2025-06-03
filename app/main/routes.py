from flask import render_template, redirect, url_for, request
from flask_login import login_required, current_user
from app.main import bp
from app.models import School, Program, Survey, SurveyResponse, Favorite, Recommendation
from app import db

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