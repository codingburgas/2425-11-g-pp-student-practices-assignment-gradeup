import os
from dotenv import load_dotenv
import platform
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    
    # Database configuration
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    if DATABASE_URL:
        # Use the provided DATABASE_URL (for production/Azure)
        if DATABASE_URL.startswith('postgres://'):
            # Fix for newer versions of SQLAlchemy with PostgreSQL
            DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        SQLALCHEMY_DATABASE_URI = DATABASE_URL
    elif platform.system() == 'Windows':
        # Local Windows development - try PostgreSQL first, then SQL Server
        SQLALCHEMY_DATABASE_URI = os.environ.get('LOCAL_DATABASE_URL') or \
            'postgresql://postgres:password@localhost/schoolrecommendation' or \
            'mssql+pyodbc://@localhost/SchoolRecommendation?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
    else:
        # Local Linux/Mac development - try PostgreSQL first, then SQL Server
        SQLALCHEMY_DATABASE_URI = os.environ.get('LOCAL_DATABASE_URL') or \
            'postgresql://postgres:password@localhost/schoolrecommendation' or \
            f"mssql+pyodbc://sa:yo(!)urStrongPassword12@localhost/SchoolRecommendation?driver=ODBC+Driver+17+for+SQL+Server"
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_SUBJECT_PREFIX = '[School Recommendation]'
    MAIL_SENDER = 'School Recommendation Admin <noreply@schoolrecommendation.com>' 