import os
import platform
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'hard-to-guess-string')

    # Email verification toggle
    DISABLE_EMAIL_VERIFICATION = os.environ.get('DISABLE_EMAIL_VERIFICATION', 'false').lower() in ['true', 'on', '1', 'yes']
    
    # Database configuration for Render (PostgreSQL)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql+psycopg://", 1)
    
    # Fallback to local database configuration if DATABASE_URL is not set
    if not SQLALCHEMY_DATABASE_URI:
        DB_USER = os.environ.get('DB_USER')
        DB_PASSWORD = os.environ.get('DB_PASSWORD')
        DB_HOST = os.environ.get('DB_HOST')
        DB_NAME = os.environ.get('DB_NAME')

        # Platform-specific database driver and connection string
        if platform.system() == 'Darwin':  # macOS
            DB_DRIVER = os.environ.get('DB_DRIVER', 'ODBC Driver 17 for SQL Server')
            SQLALCHEMY_DATABASE_URI = (
                f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
                f"?driver={DB_DRIVER.replace(' ', '+')}"
                f"&TrustServerCertificate=yes"
                f"&Encrypt=no"
            )
        else:  # Windows/Linux
            SQLALCHEMY_DATABASE_URI = 'mssql+pyodbc://@localhost/SchoolRecommendation?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_SUBJECT_PREFIX = '[School Recommendation]'
    MAIL_SENDER = os.environ.get('MAIL_USERNAME')  # Use the authenticated Gmail account as sender
