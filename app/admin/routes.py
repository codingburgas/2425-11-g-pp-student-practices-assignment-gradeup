from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from app.admin import bp
from app.models import User, School, Program, Survey, SurveyResponse, Recommendation, Favorite
from app import db
import json
from app.admin.forms import EditUserForm

def admin_required(f):
    """Decorator for requiring admin access to a route"""
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('You need admin privileges to access this page.', 'danger')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@bp.route('/dashboard')
@admin_required
def dashboard():
    # Get dashboard statistics
    stats = {
        'total_users': User.query.count(),
        'total_universities': School.query.count(),
        'total_programs': Program.query.count(),
        'total_surveys': Survey.query.count(),
        'active_surveys': Survey.query.filter_by(is_active=True).count(),
        'total_responses': SurveyResponse.query.count(),
        'admin_users': User.query.filter_by(is_admin=True).count(),
        'recent_users': User.query.order_by(User.created_at.desc()).limit(5).all(),
        'recent_responses': SurveyResponse.query.order_by(SurveyResponse.created_at.desc()).limit(5).all()
    }
    return render_template('admin/dashboard.html', title='Admin Dashboard', stats=stats)

@bp.route('/users')
@admin_required
def users():
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '', type=str)
    
    query = User.query
    if search:
        query = query.filter(User.username.contains(search) | User.email.contains(search))
    
    users = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    return render_template('admin/users.html', title='User Management', users=users, search=search)

@bp.route('/users/<int:user_id>')
@admin_required
def user_detail(user_id):
    user = User.query.get_or_404(user_id)
    return render_template('admin/user_detail.html', title=f'User: {user.username}', user=user)

@bp.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    form = EditUserForm(original_user=user)
    
    if form.validate_on_submit():
        user.username = form.username.data
        user.email = form.email.data
        user.bio = form.bio.data
        user.location = form.location.data
        
        # Only allow changing admin status if not editing own account
        if user != current_user:
            user.is_admin = form.is_admin.data
        
        try:
            db.session.commit()
            flash(f'User {user.username} updated successfully.', 'success')
            return redirect(url_for('admin.users'))
        except Exception as e:
            db.session.rollback()
            flash('Error updating user.', 'danger')
    
    # Pre-populate form with user data
    if request.method == 'GET':
        form.username.data = user.username
        form.email.data = user.email
        form.bio.data = user.bio
        form.location.data = user.location
        form.is_admin.data = user.is_admin
    
    return render_template('admin/edit_user.html', title=f'Edit User: {user.username}', user=user, form=form)

@bp.route('/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if user == current_user:
        flash('You cannot delete your own account.', 'warning')
        return redirect(url_for('admin.users'))
    
    try:
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.username} deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error deleting user.', 'danger')
    
    return redirect(url_for('admin.users'))

@bp.route('/toggle_admin/<int:user_id>')
@admin_required
def toggle_admin(user_id):
    user = User.query.get_or_404(user_id)
    if user == current_user:
        flash('You cannot modify your own admin status.', 'warning')
    else:
        user.is_admin = not user.is_admin
        db.session.commit()
        status = 'granted' if user.is_admin else 'revoked'
        flash(f'Admin privileges {status} for {user.username}.', 'success')
    return redirect(url_for('admin.users'))

# University Management
@bp.route('/universities')
@admin_required
def admin_universities():
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '', type=str)
    
    query = School.query
    if search:
        query = query.filter(School.name.contains(search) | School.location.contains(search))
    
    universities = query.order_by(School.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    return render_template('admin/universities.html', title='University Management', universities=universities, search=search)

@bp.route('/universities/new', methods=['GET', 'POST'])
@admin_required
def new_university():
    if request.method == 'POST':
        university = School(
            name=request.form.get('name'),
            description=request.form.get('description'),
            location=request.form.get('location'),
            website=request.form.get('website'),
            email=request.form.get('email'),
            phone=request.form.get('phone'),
            admission_requirements=request.form.get('admission_requirements')
        )
        
        try:
            db.session.add(university)
            db.session.commit()
            flash(f'University {university.name} created successfully.', 'success')
            return redirect(url_for('admin.admin_universities'))
        except Exception as e:
            db.session.rollback()
            flash('Error creating university.', 'danger')
    
    return render_template('admin/edit_university.html', title='New University', university=None)

@bp.route('/universities/<int:university_id>')
@admin_required
def university_detail(university_id):
    university = School.query.get_or_404(university_id)
    return render_template('admin/university_detail.html', title=f'University: {university.name}', university=university)

@bp.route('/universities/<int:university_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_university(university_id):
    university = School.query.get_or_404(university_id)
    
    if request.method == 'POST':
        university.name = request.form.get('name', university.name)
        university.description = request.form.get('description', university.description)
        university.location = request.form.get('location', university.location)
        university.website = request.form.get('website', university.website)
        university.email = request.form.get('email', university.email)
        university.phone = request.form.get('phone', university.phone)
        university.admission_requirements = request.form.get('admission_requirements', university.admission_requirements)
        
        try:
            db.session.commit()
            flash(f'University {university.name} updated successfully.', 'success')
            return redirect(url_for('admin.admin_universities'))
        except Exception as e:
            db.session.rollback()
            flash('Error updating university.', 'danger')
    
    return render_template('admin/edit_university.html', title=f'Edit University: {university.name}', university=university)

@bp.route('/universities/<int:university_id>/delete', methods=['POST'])
@admin_required
def delete_university(university_id):
    university = School.query.get_or_404(university_id)
    
    try:
        db.session.delete(university)
        db.session.commit()
        flash(f'University {university.name} deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error deleting university.', 'danger')
    
    return redirect(url_for('admin.admin_universities'))

# Program Management
@bp.route('/programs')
@admin_required
def programs():
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '', type=str)
    university_id = request.args.get('university_id', None, type=int)
    
    query = Program.query.join(School)
    if search:
        query = query.filter(Program.name.contains(search) | Program.degree_type.contains(search))
    if university_id:
        query = query.filter(Program.school_id == university_id)
    
    programs = query.order_by(Program.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    universities = School.query.all()
    return render_template('admin/programs.html', title='Program Management', programs=programs, universities=universities, search=search, selected_university=university_id)

@bp.route('/programs/new', methods=['GET', 'POST'])
@admin_required
def new_program():
    if request.method == 'POST':
        program = Program(
            name=request.form.get('name'),
            description=request.form.get('description'),
            duration=request.form.get('duration'),
            degree_type=request.form.get('degree_type'),
            admission_requirements=request.form.get('admission_requirements'),
            tuition_fee=float(request.form.get('tuition_fee', 0)) if request.form.get('tuition_fee') else None,
            school_id=int(request.form.get('school_id'))
        )
        
        try:
            db.session.add(program)
            db.session.commit()
            flash(f'Program {program.name} created successfully.', 'success')
            return redirect(url_for('admin.programs'))
        except Exception as e:
            db.session.rollback()
            flash('Error creating program.', 'danger')
    
    universities = School.query.all()
    return render_template('admin/edit_program.html', title='New Program', program=None, universities=universities)

@bp.route('/programs/<int:program_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_program(program_id):
    program = Program.query.get_or_404(program_id)
    
    if request.method == 'POST':
        program.name = request.form.get('name', program.name)
        program.description = request.form.get('description', program.description)
        program.duration = request.form.get('duration', program.duration)
        program.degree_type = request.form.get('degree_type', program.degree_type)
        program.admission_requirements = request.form.get('admission_requirements', program.admission_requirements)
        program.tuition_fee = float(request.form.get('tuition_fee', 0)) if request.form.get('tuition_fee') else program.tuition_fee
        program.school_id = int(request.form.get('school_id'))
        
        try:
            db.session.commit()
            flash(f'Program {program.name} updated successfully.', 'success')
            return redirect(url_for('admin.programs'))
        except Exception as e:
            db.session.rollback()
            flash('Error updating program.', 'danger')
    
    universities = School.query.all()
    return render_template('admin/edit_program.html', title=f'Edit Program: {program.name}', program=program, universities=universities)

@bp.route('/programs/<int:program_id>/delete', methods=['POST'])
@admin_required
def delete_program(program_id):
    program = Program.query.get_or_404(program_id)
    
    try:
        db.session.delete(program)
        db.session.commit()
        flash(f'Program {program.name} deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error deleting program.', 'danger')
    
    return redirect(url_for('admin.programs'))

# Survey Management
@bp.route('/surveys')
@admin_required
def surveys():
    page = request.args.get('page', 1, type=int)
    surveys = Survey.query.order_by(Survey.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    return render_template('admin/surveys.html', title='Survey Management', surveys=surveys)

@bp.route('/surveys/new', methods=['GET', 'POST'])
@admin_required
def new_survey():
    if request.method == 'POST':
        questions_json = request.form.get('questions')
        try:
            questions = json.loads(questions_json)
        except json.JSONDecodeError:
            flash('Invalid JSON format for questions.', 'danger')
            return render_template('admin/edit_survey.html', title='New Survey', survey=None)
        
        survey = Survey(
            title=request.form.get('title'),
            description=request.form.get('description'),
            questions=questions_json,
            is_active=bool(request.form.get('is_active'))
        )
        
        try:
            db.session.add(survey)
            db.session.commit()
            flash(f'Survey "{survey.title}" created successfully.', 'success')
            return redirect(url_for('admin.surveys'))
        except Exception as e:
            db.session.rollback()
            flash('Error creating survey.', 'danger')
    
    return render_template('admin/edit_survey.html', title='New Survey', survey=None)

@bp.route('/surveys/<int:survey_id>')
@admin_required
def survey_detail(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    
    # Get the latest response (most recent first)
    latest_response = survey.responses.order_by(SurveyResponse.created_at.desc()).first() if survey.responses.count() > 0 else None
    
    # Get recent responses (last 5, most recent first)
    recent_responses = survey.responses.order_by(SurveyResponse.created_at.desc()).limit(5).all()
    
    return render_template('admin/survey_detail.html', 
                         title=f'Survey: {survey.title}', 
                         survey=survey,
                         latest_response=latest_response,
                         recent_responses=recent_responses)

@bp.route('/surveys/<int:survey_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_survey(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    
    if request.method == 'POST':
        questions_json = request.form.get('questions')
        try:
            questions = json.loads(questions_json)
        except json.JSONDecodeError:
            flash('Invalid JSON format for questions.', 'danger')
            return render_template('admin/edit_survey.html', title=f'Edit Survey: {survey.title}', survey=survey)
        
        survey.title = request.form.get('title', survey.title)
        survey.description = request.form.get('description', survey.description)
        survey.questions = questions_json
        survey.is_active = bool(request.form.get('is_active'))
        
        try:
            db.session.commit()
            flash(f'Survey "{survey.title}" updated successfully.', 'success')
            return redirect(url_for('admin.surveys'))
        except Exception as e:
            db.session.rollback()
            flash('Error updating survey.', 'danger')
    
    return render_template('admin/edit_survey.html', title=f'Edit Survey: {survey.title}', survey=survey)

@bp.route('/surveys/<int:survey_id>/delete', methods=['POST'])
@admin_required
def delete_survey(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    
    try:
        db.session.delete(survey)
        db.session.commit()
        flash(f'Survey "{survey.title}" deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error deleting survey.', 'danger')
    
    return redirect(url_for('admin.surveys'))

@bp.route('/surveys/<int:survey_id>/toggle_active')
@admin_required
def toggle_survey_active(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    survey.is_active = not survey.is_active
    
    try:
        db.session.commit()
        status = 'activated' if survey.is_active else 'deactivated'
        flash(f'Survey "{survey.title}" {status} successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error updating survey status.', 'danger')
    
    return redirect(url_for('admin.surveys'))

# Survey Responses
@bp.route('/survey-responses')
@admin_required
def survey_responses():
    page = request.args.get('page', 1, type=int)
    survey_id = request.args.get('survey_id', None, type=int)
    
    query = SurveyResponse.query.join(User).join(Survey)
    if survey_id:
        query = query.filter(SurveyResponse.survey_id == survey_id)
    
    responses = query.order_by(SurveyResponse.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    surveys = Survey.query.all()
    return render_template('admin/survey_responses.html', title='Survey Responses', responses=responses, surveys=surveys, selected_survey=survey_id) 