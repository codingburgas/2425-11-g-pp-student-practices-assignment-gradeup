from flask import render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from app.admin import bp
from app.models import User
from app import db

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
    users = User.query.all()
    return render_template('admin/dashboard.html', title='Admin Dashboard', users=users)

@bp.route('/users')
@admin_required
def users():
    users = User.query.all()
    return render_template('admin/users.html', title='User Management', users=users)

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