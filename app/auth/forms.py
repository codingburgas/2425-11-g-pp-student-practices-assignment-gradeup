from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, SelectMultipleField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from app.models import User

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email(), Length(max=120)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    password2 = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')

class ResetPasswordRequestForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired(), Length(min=8)])
    password2 = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

class ProfileForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(), 
        Length(min=2, max=64)
    ])
    email = StringField('Email', validators=[
        DataRequired(), 
        Email()
    ])
    bio = TextAreaField('Bio', validators=[Length(max=500)])
    location = StringField('Location', validators=[Length(max=100)])
    picture = FileField('Update Profile Picture', validators=[FileAllowed(['jpg', 'png', 'jpeg'])])
    submit = SubmitField('Update Profile')

class UserPreferencesForm(FlaskForm):
    preferred_degree_types = SelectMultipleField('Preferred Degree Types', 
        choices=[
            ('bachelors', "Bachelor's"),
            ('masters', "Master's"),
            ('phd', 'PhD'),
            ('associate', "Associate's"),
            ('certificate', 'Certificate')
        ])
    preferred_locations = TextAreaField('Preferred Locations', 
        validators=[Length(max=500)],
        description='Enter preferred locations, one per line')
    max_tuition = StringField('Maximum Tuition Fee', 
        validators=[Length(max=20)],
        description='Enter maximum tuition fee you can afford')
    preferred_programs = TextAreaField('Preferred Programs', 
        validators=[Length(max=500)],
        description='Enter preferred programs or fields of study, one per line')
    submit = SubmitField('Save Preferences') 