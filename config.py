import os
from dotenv import load_dotenv
import platform
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    
    
    
    if platform.system() == 'Windows':
        SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
            'mssql+pyodbc://@localhost/SchoolRecommendation?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
        
        SQLALCHEMY_TRACK_MODIFICATIONS = False
    else:
         SQLALCHEMY_DATABASE_URI = f"mssql+pyodbc://sa:yo(!)urStrongPassword12@localhost/SchoolRecommendation?driver=ODBC+Driver+17+for+SQL+Server"
    
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_SUBJECT_PREFIX = '[School Recommendation]'
    MAIL_SENDER = 'School Recommendation Admin <noreply@schoolrecommendation.com>' 