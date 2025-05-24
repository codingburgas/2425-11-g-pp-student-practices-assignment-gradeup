from flask import render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, current_user, login_required
from urllib.parse import urlparse
from werkzeug.utils import secure_filename
import os
from app import db
from app.auth import bp
from app.auth.forms import LoginForm, RegistrationForm, ResetPasswordRequestForm, ResetPasswordForm, ProfileForm
from app.models import User

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid email or password', 'danger')
            return render_template('auth/login.html', title='Sign In', form=form)
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or urlparse(next_page).netloc != '':
            next_page = url_for('main.dashboard')
        flash('You are now logged in!', 'success')
        return redirect(next_page)
    return render_template('auth/login.html', title='Sign In', form=form)

@bp.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!', 'success')
        return redirect(url_for('auth.login'))
    return render_template('auth/register.html', title='Register', form=form)

@bp.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = ResetPasswordRequestForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            # For now, we'll just simulate the password reset
            flash('Check your email for the instructions to reset your password', 'info')
            return redirect(url_for('auth.login'))
        else:
            flash('Email not found in our records', 'warning')
            return render_template('auth/reset_password_request.html', title='Reset Password', form=form)
    return render_template('auth/reset_password_request.html', title='Reset Password', form=form)

@bp.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    # For now, we'll just simulate token verification
    user = User.query.first()  # This would normally verify the token
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
        # Check if username or email is being changed
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
        
        # Handle profile picture upload
        if form.profile_picture.data:
            # Create uploads directory if it doesn't exist
            uploads_dir = os.path.join('app', 'static', 'uploads')
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)
            
            # Save the file
            f = form.profile_picture.data
            filename = secure_filename(f.filename)
            # Add user id to filename to make it unique
            filename = f"{current_user.id}_{filename}"
            filepath = os.path.join(uploads_dir, filename)
            f.save(filepath)
            current_user.profile_picture = filename
        
        # Update user fields
        current_user.username = form.username.data
        current_user.email = form.email.data
        current_user.bio = form.bio.data
        current_user.location = form.location.data
        current_user.phone = form.phone.data
        current_user.date_of_birth = form.date_of_birth.data
        
        db.session.commit()
        flash('Your profile has been updated.', 'success')
        return redirect(url_for('auth.profile'))
    
    # Pre-populate form with current user data
    elif request.method == 'GET':
        form.bio.data = current_user.bio
        form.location.data = current_user.location
        form.phone.data = current_user.phone
        form.date_of_birth.data = current_user.date_of_birth
    
    return render_template('auth/profile.html', title='Profile', form=form)

@bp.route('/update_preferences', methods=['POST'])
@login_required
def update_preferences():
    """Update user preferences via AJAX"""
    try:
        preferences = request.get_json()
        current_user.set_preferences(preferences)
        db.session.commit()
        return {'status': 'success', 'message': 'Preferences updated successfully'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 400 