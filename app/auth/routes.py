from flask import render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, current_user, login_required
from urllib.parse import urlparse
from app import db
from app.auth import bp
from app.auth.forms import LoginForm, RegistrationForm, ResetPasswordRequestForm, ResetPasswordForm, ProfileForm, UserPreferencesForm, ChangePasswordForm
from app.models import User
import os
from werkzeug.utils import secure_filename
from flask import current_app

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.survey'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            if not next_page or urlparse(next_page).netloc != '':
                next_page = url_for('main.survey')
            
            return redirect(next_page)
        
        
        return render_template('auth/login.html', title='Sign In', form=form, 
                             error='Invalid email or password')
    
    return render_template('auth/login.html', title='Sign In', form=form)

@bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('main.index'))

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.survey'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        
        
        login_user(user)
        return redirect(url_for('main.survey'))
    
    return render_template('auth/register.html', title='Register', form=form)

@bp.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = ResetPasswordRequestForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            
            flash('Check your email for the instructions to reset your password', 'info')
            return redirect(url_for('auth.login'))
        else:
            flash('Email not found in our records', 'warning')
    return render_template('auth/reset_password_request.html', title='Reset Password', form=form)

@bp.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    user = User.query.first()  
    if not user:
        flash('Invalid or expired token', 'danger')
        return redirect(url_for('auth.reset_password_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash('Your password has been reset.', 'success')
        return redirect(url_for('auth.login'))
    return render_template('auth/reset_password.html', title='Reset Password', form=form)

@bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = ProfileForm(obj=current_user)
    if form.validate_on_submit():
        
        if form.username.data != current_user.username:
            existing_user = User.query.filter_by(username=form.username.data).first()
            if existing_user:
                flash('Username already in use.', 'danger')
                return render_template('auth/profile.html', title='Profile', form=form)
        
        if form.email.data != current_user.email:
            existing_user = User.query.filter_by(email=form.email.data).first()
            if existing_user:
                flash('Email already in use.', 'danger')
                return render_template('auth/profile.html', title='Profile', form=form)
        
        current_user.username = form.username.data
        current_user.email = form.email.data
        current_user.bio = form.bio.data
        current_user.location = form.location.data
        
        
        if form.picture.data:
            try:
                picture_file = save_picture(form.picture.data)
                
                if current_user.profile_picture:
                    old_picture_path = os.path.join(current_app.root_path, 'static', 'profile_pics', current_user.profile_picture)
                    if os.path.exists(old_picture_path):
                        os.remove(old_picture_path)
                current_user.profile_picture = picture_file
            except ValueError as e:
                flash(str(e), 'danger')
                return render_template('auth/profile.html', title='Profile', form=form)
            except Exception as e:
                flash('Error uploading profile picture. Please try again.', 'danger')
                return render_template('auth/profile.html', title='Profile', form=form)
        
        db.session.commit()
        flash('Your profile has been updated.', 'success')
        return redirect(url_for('auth.profile'))
    
    return render_template('auth/profile.html', title='Profile', form=form)

def save_picture(form_picture):
    """Save uploaded picture to static/profile_pics directory"""
    
    if len(form_picture.read()) > 2 * 1024 * 1024:  
        form_picture.seek(0)  
        raise ValueError('File size must be less than 2MB')
    
    
    form_picture.seek(0)
    
    
    random_hex = os.urandom(8).hex()
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    
    
    picture_path = os.path.join(current_app.root_path, 'static', 'profile_pics')
    if not os.path.exists(picture_path):
        os.makedirs(picture_path)
    
    
    picture_path = os.path.join(picture_path, picture_fn)
    form_picture.save(picture_path)
    
    return picture_fn

@bp.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    form = ChangePasswordForm()
    if form.validate_on_submit():
        current_user.set_password(form.new_password.data)
        db.session.commit()
        flash('Your password has been changed successfully!', 'success')
        return redirect(url_for('auth.profile'))
    
    return render_template('auth/change_password.html', title='Change Password', form=form)

@bp.route('/preferences', methods=['GET', 'POST'])
@login_required
def preferences():
    form = UserPreferencesForm()
    
    
    if request.method == 'GET':
        prefs = current_user.get_preferences()
        form.preferred_degree_types.data = prefs.get('degree_types', [])
        form.preferred_locations.data = '\n'.join(prefs.get('locations', []))
        form.max_tuition.data = prefs.get('max_tuition', '')
        form.preferred_programs.data = '\n'.join(prefs.get('programs', []))
    
    if form.validate_on_submit():
        
        prefs = {
            'degree_types': form.preferred_degree_types.data,
            'locations': [loc.strip() for loc in form.preferred_locations.data.split('\n') if loc.strip()],
            'max_tuition': form.max_tuition.data,
            'programs': [prog.strip() for prog in form.preferred_programs.data.split('\n') if prog.strip()]
        }
        
        
        current_user.set_preferences(prefs)
        db.session.commit()
        flash('Your preferences have been saved.', 'success')
        return redirect(url_for('auth.preferences'))
    
    return render_template('auth/preferences.html', title='Preferences', form=form) 