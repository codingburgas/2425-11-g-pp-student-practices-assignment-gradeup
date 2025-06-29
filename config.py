import os
import platform
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # Email verification toggle
    DISABLE_EMAIL_VERIFICATION = os.environ.get('DISABLE_EMAIL_VERIFICATION', 'false').lower() in ['true', 'on', '1', 'yes']
    
    # Use the new PostgreSQL database URL
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://gradeup_user:lEKw2auoDCQe8ZlEzRBGam9agzqbs56f@dpg-d1go5afgi27c73bv8vb0-a/gradeup_db_uopn_asj1'
    
    # Handle SSL for PostgreSQL on Render
    if SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)
    
    # Add SSL mode for PostgreSQL
    if 'postgresql://' in SQLALCHEMY_DATABASE_URI:
        if '?' not in SQLALCHEMY_DATABASE_URI:
            SQLALCHEMY_DATABASE_URI += '?sslmode=require'
        elif 'sslmode=' not in SQLALCHEMY_DATABASE_URI:
            SQLALCHEMY_DATABASE_URI += '&sslmode=require'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAIL_SERVER = os.environ.get('MAIL_SERVER') or 'smtp.gmail.com'
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_SUBJECT_PREFIX = '[School Recommendation]'
    MAIL_SENDER = os.environ.get('MAIL_USERNAME')  # Use the authenticated Gmail account as sender

    # Application settings
    POSTS_PER_PAGE = 10
    LANGUAGES = ['en', 'bg']
    
    # Email verification settings
    EMAIL_VERIFICATION_REQUIRED = os.environ.get('EMAIL_VERIFICATION_REQUIRED', 'false').lower() == 'true'
