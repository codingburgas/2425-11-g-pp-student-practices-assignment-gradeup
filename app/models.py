from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db, login
import json
import secrets

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(400))
    is_admin = db.Column(db.Boolean, default=False)
    email_verified = db.Column(db.Boolean, default=False)
    email_verification_token = db.Column(db.String(100), nullable=True)
    email_verification_token_expires = db.Column(db.DateTime, nullable=True)
    password_reset_token = db.Column(db.String(100), nullable=True)
    password_reset_token_expires = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    profile_picture = db.Column(db.String(255), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    location = db.Column(db.String(100), nullable=True)
    preferences = db.Column(db.Text, nullable=True)  
    
    
    survey_responses = db.relationship('SurveyResponse', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    
    favorites = db.relationship('Favorite', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

    def get_preferences(self):
        """Get user preferences as a dictionary"""
        if self.preferences:
            return json.loads(self.preferences)
        return {}
    
    def set_preferences(self, prefs_dict):
        """Set user preferences from a dictionary"""
        self.preferences = json.dumps(prefs_dict)
    
    def generate_email_verification_token(self):
        """Generate a new email verification token"""
        self.email_verification_token = secrets.token_urlsafe(32)
        self.email_verification_token_expires = datetime.utcnow() + timedelta(hours=24)
        return self.email_verification_token
    
    def verify_email_token(self, token):
        """Verify email verification token"""
        if (self.email_verification_token == token and 
            self.email_verification_token_expires and
            self.email_verification_token_expires > datetime.utcnow()):
            self.email_verified = True
            self.email_verification_token = None
            self.email_verification_token_expires = None
            return True
        return False
    
    def is_email_verification_token_expired(self):
        """Check if email verification token is expired"""
        if not self.email_verification_token_expires:
            return True
        return self.email_verification_token_expires < datetime.utcnow()
    
    def generate_password_reset_token(self):
        """Generate a new password reset token"""
        self.password_reset_token = secrets.token_urlsafe(32)
        self.password_reset_token_expires = datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
        return self.password_reset_token
    
    def verify_password_reset_token(self, token):
        """Verify password reset token"""
        if (self.password_reset_token == token and 
            self.password_reset_token_expires and
            self.password_reset_token_expires > datetime.utcnow()):
            return True
        return False
    
    def clear_password_reset_token(self):
        """Clear password reset token after successful reset"""
        self.password_reset_token = None
        self.password_reset_token_expires = None
    
    def is_password_reset_token_expired(self):
        """Check if password reset token is expired"""
        if not self.password_reset_token_expires:
            return True
        return self.password_reset_token_expires < datetime.utcnow()
    
    def get_public_profile(self):
        """Get publicly visible profile information"""
        return {
            'id': self.id,
            'username': self.username,
            'bio': self.bio,
            'location': self.location,
            'created_at': self.created_at,
            'profile_picture': self.profile_picture,
            'survey_count': self.survey_responses.count(),
            'favorites_count': self.favorites.count()
        }
    
    def can_view_profile(self, viewer=None):
        """Check if a user can view this profile"""
        # For now, all profiles are public
        # Can be extended for privacy controls later
        return True
    
    def get_display_name(self):
        """Get display name for the user"""
        return self.username

@login.user_loader
def load_user(id):
    return User.query.get(int(id))

class School(db.Model):
    __tablename__ = 'schools'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    location = db.Column(db.String(100), nullable=False)
    website = db.Column(db.String(255), nullable=True)
    email = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    logo = db.Column(db.String(255), nullable=True)
    admission_requirements = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    
    programs = db.relationship('Program', backref='school', lazy='dynamic', cascade='all, delete-orphan')
    
    
    favorites = db.relationship('Favorite', backref='school', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<School {self.name}>'
    
class Program(db.Model):
    __tablename__ = 'programs'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    duration = db.Column(db.String(50), nullable=True)  
    degree_type = db.Column(db.String(50), nullable=False)  
    admission_requirements = db.Column(db.Text, nullable=True)
    tuition_fee = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    
    school_id = db.Column(db.Integer, db.ForeignKey('schools.id'), nullable=False)
    
    
    recommendations = db.relationship('Recommendation', backref='program', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Program {self.name} at {self.school.name}>'

class Survey(db.Model):
    __tablename__ = 'surveys'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    questions = db.Column(db.Text, nullable=False)  
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    
    responses = db.relationship('SurveyResponse', backref='survey', lazy='dynamic', cascade='all, delete-orphan')
    
    def get_questions(self):
        return json.loads(self.questions)
    
    def set_questions(self, questions_list):
        self.questions = json.dumps(questions_list)
    
    def __repr__(self):
        return f'<Survey {self.title}>'

class SurveyResponse(db.Model):
    __tablename__ = 'survey_responses'
    
    id = db.Column(db.Integer, primary_key=True)
    answers = db.Column(db.Text, nullable=False)  
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    survey_id = db.Column(db.Integer, db.ForeignKey('surveys.id'), nullable=False)
    
    
    recommendations = db.relationship('Recommendation', backref='survey_response', lazy='dynamic', cascade='all, delete-orphan')
    
    def get_answers(self):
        return json.loads(self.answers)
    
    def set_answers(self, answers_dict):
        self.answers = json.dumps(answers_dict)
    
    def __repr__(self):
        return f'<SurveyResponse {self.id} by User {self.user_id}>'

class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    
    id = db.Column(db.Integer, primary_key=True)
    score = db.Column(db.Float, nullable=False)  
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    
    survey_response_id = db.Column(db.Integer, db.ForeignKey('survey_responses.id'), nullable=False)
    program_id = db.Column(db.Integer, db.ForeignKey('programs.id'), nullable=False)
    
    def __repr__(self):
        return f'<Recommendation {self.id} for Program {self.program_id} with score {self.score}>'

class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    survey_response_id = db.Column(db.Integer, db.ForeignKey('survey_responses.id'), nullable=True)
    
    # Prediction input data (JSON stored as text)
    input_features = db.Column(db.Text, nullable=False)
    
    # Prediction results
    predictions = db.Column(db.Text, nullable=False)  # JSON array of predictions
    confidence_scores = db.Column(db.Text, nullable=False)  # JSON array of confidence scores
    model_version = db.Column(db.String(50), nullable=True)
    
    # Metadata
    prediction_type = db.Column(db.String(50), default='individual')  # 'individual' or 'batch'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('prediction_history', lazy='dynamic'))
    survey_response = db.relationship('SurveyResponse', backref=db.backref('prediction_history', lazy='dynamic'))
    
    def get_input_features(self):
        """Get input features as a dictionary"""
        return json.loads(self.input_features)
    
    def set_input_features(self, features_dict):
        """Set input features from a dictionary"""
        self.input_features = json.dumps(features_dict)
    
    def get_predictions(self):
        """Get predictions as a list"""
        return json.loads(self.predictions)
    
    def set_predictions(self, predictions_list):
        """Set predictions from a list"""
        self.predictions = json.dumps(predictions_list)
    
    def get_confidence_scores(self):
        """Get confidence scores as a list"""
        return json.loads(self.confidence_scores)
    
    def set_confidence_scores(self, scores_list):
        """Set confidence scores from a list"""
        self.confidence_scores = json.dumps(scores_list)
    
    def __repr__(self):
        return f'<PredictionHistory {self.id} for User {self.user_id}>'

class Favorite(db.Model):
    __tablename__ = 'favorites'
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    school_id = db.Column(db.Integer, db.ForeignKey('schools.id'), nullable=False)
    
    __table_args__ = (db.UniqueConstraint('user_id', 'school_id', name='unique_user_school_favorite'),)
    
    def __repr__(self):
        return f'<Favorite {self.id} by User {self.user_id} for School {self.school_id}>' 