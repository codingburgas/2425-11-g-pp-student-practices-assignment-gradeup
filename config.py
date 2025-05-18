import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

# Update this to match your SQL Server instance
# For default instance use: 'localhost' or '.'
# For named instance use: 'localhost\\INSTANCENAME' or '.\\SQLEXPRESS'
SQL_SERVER_INSTANCE = 'localhost'

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    
    # SQL Server connection string with Windows Authentication
    # Format for Windows Authentication: 'mssql+pyodbc:///?odbc_connect=DRIVER={ODBC Driver 17 for SQL Server};SERVER=servername;DATABASE=SchoolRecommendation;Trusted_Connection=yes'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        f'mssql+pyodbc:///?odbc_connect=DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SQL_SERVER_INSTANCE};DATABASE=SchoolRecommendation;Trusted_Connection=yes'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Mail settings
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_SUBJECT_PREFIX = '[School Recommendation]'
    MAIL_SENDER = 'School Recommendation Admin <noreply@schoolrecommendation.com>' 