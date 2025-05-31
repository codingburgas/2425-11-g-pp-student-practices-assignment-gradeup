from flask import render_template, redirect, url_for, request
from flask_login import login_required, current_user
from app.main import bp
from app.models import School, Program, Survey

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