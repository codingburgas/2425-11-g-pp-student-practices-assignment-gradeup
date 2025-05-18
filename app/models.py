from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db, login
import json

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(400))
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    profile_picture = db.Column(db.String(255), nullable=True)
    
    # One-to-many relationship with survey responses
    survey_responses = db.relationship('SurveyResponse', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    # One-to-many relationship with favorite schools
    favorites = db.relationship('Favorite', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

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
    
    # One-to-many relationship with programs
    programs = db.relationship('Program', backref='school', lazy='dynamic', cascade='all, delete-orphan')
    
    # One-to-many relationship with favorites
    favorites = db.relationship('Favorite', backref='school', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<School {self.name}>'
    
class Program(db.Model):
    __tablename__ = 'programs'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    duration = db.Column(db.String(50), nullable=True)  # e.g., "4 years"
    degree_type = db.Column(db.String(50), nullable=False)  # e.g., "Bachelor's", "Master's"
    admission_requirements = db.Column(db.Text, nullable=True)
    tuition_fee = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign key
    school_id = db.Column(db.Integer, db.ForeignKey('schools.id'), nullable=False)
    
    # Many-to-many relationship with survey responses through recommendations
    recommendations = db.relationship('Recommendation', backref='program', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Program {self.name} at {self.school.name}>'

class Survey(db.Model):
    __tablename__ = 'surveys'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    questions = db.Column(db.Text, nullable=False)  # JSON string of questions
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # One-to-many relationship with survey responses
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
    answers = db.Column(db.Text, nullable=False)  # JSON string of answers
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    survey_id = db.Column(db.Integer, db.ForeignKey('surveys.id'), nullable=False)
    
    # One-to-many relationship with recommendations
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
    score = db.Column(db.Float, nullable=False)  # Matching score between survey response and program
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    survey_response_id = db.Column(db.Integer, db.ForeignKey('survey_responses.id'), nullable=False)
    program_id = db.Column(db.Integer, db.ForeignKey('programs.id'), nullable=False)
    
    def __repr__(self):
        return f'<Recommendation {self.id} for Program {self.program_id} with score {self.score}>'

class Favorite(db.Model):
    __tablename__ = 'favorites'
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    school_id = db.Column(db.Integer, db.ForeignKey('schools.id'), nullable=False)
    
    __table_args__ = (db.UniqueConstraint('user_id', 'school_id', name='unique_user_school_favorite'),)
    
    def __repr__(self):
        return f'<Favorite {self.id} by User {self.user_id} for School {self.school_id}>' 