"""
Azure App Service startup file for GradeUP Flask Application
"""
import os
from app import create_app, db
from app.models import User, School, Program, Survey, SurveyResponse

# Create the Flask application
app = create_app()

# Initialize database on startup (for Azure)
with app.app_context():
    try:
        # Create all database tables
        db.create_all()
        print("Database tables created successfully!")
    except Exception as e:
        print(f"Database initialization error: {str(e)}")

if __name__ == "__main__":
    # For local development
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 