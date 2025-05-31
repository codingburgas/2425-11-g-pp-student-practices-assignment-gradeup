from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Email, Length, ValidationError
from wtforms.widgets import html_params
from markupsafe import Markup
from app.models import User
from flask_login import current_user

class EditUserForm(FlaskForm):
    username = StringField(Markup('<i class="fas fa-user"></i> Username'), validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField(Markup('<i class="fas fa-envelope"></i> Email Address'), validators=[DataRequired(), Email(), Length(max=120)])
    bio = TextAreaField(Markup('<i class="fas fa-quote-left"></i> Biography'), validators=[Length(max=500)])
    location = StringField(Markup('<i class="fas fa-map-marker-alt"></i> Location'), validators=[Length(max=100)])
    is_admin = BooleanField(Markup('<span class="checkbox-custom"></span><span class="checkbox-text"><i class="fas fa-crown"></i> Grant Administrator Privileges</span>'))
    submit = SubmitField('Update User')

    def __init__(self, original_user, *args, **kwargs):
        super(EditUserForm, self).__init__(*args, **kwargs)
        self.original_user = original_user

    def validate_username(self, username):
        if username.data != self.original_user.username:
            user = User.query.filter_by(username=username.data).first()
            if user is not None:
                raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        if email.data != self.original_user.email:
            user = User.query.filter_by(email=email.data).first()
            if user is not None:
                raise ValidationError('Please use a different email address.') 