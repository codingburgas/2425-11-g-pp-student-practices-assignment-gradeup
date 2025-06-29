#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Drop all tables and recreate them with correct schema
python -c "
from app import create_app
from app.models import db, User, School, Program, Survey, SurveyResponse, Recommendation, PredictionHistory
app = create_app()
with app.app_context():
    # Drop all tables
    db.drop_all()
    print('All tables dropped')
    
    # Create all tables with correct schema
    db.create_all()
    print('All tables created successfully with correct schema')
    
    # Verify the User table has all required columns
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    columns = [col['name'] for col in inspector.get_columns('users')]
    required_columns = ['id', 'username', 'email', 'password_hash', 'is_admin', 
                       'email_verified', 'email_verification_token', 'email_verification_token_expires',
                       'password_reset_token', 'password_reset_token_expires', 'created_at',
                       'profile_picture', 'bio', 'location', 'preferences']
    
    missing_columns = [col for col in required_columns if col not in columns]
    if missing_columns:
        print(f'WARNING: Missing columns: {missing_columns}')
    else:
        print('All required columns present in users table')
" 