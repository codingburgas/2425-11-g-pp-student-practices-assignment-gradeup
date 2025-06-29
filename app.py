from app import create_app, db
from app.models import User, School, Program, Survey, SurveyResponse, Recommendation
import os

app = create_app()

# Initialize database tables on startup with better error handling
with app.app_context():
    try:
        print("🔄 Initializing database...")
        
        # Print current database URI for debugging
        db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', 'Not set')
        if 'sqlite' in db_uri.lower():
            print("✅ Using SQLite database")
        elif 'postgresql' in db_uri.lower():
            print("✅ Using PostgreSQL database")
        else:
            print(f"⚠️ Unknown database type: {db_uri}")
            
        # Create all tables if they don't exist
        db.create_all()
        print("✅ Database tables created/verified successfully")
        
        # Test a simple query
        from sqlalchemy import text
        result = db.session.execute(text('SELECT 1'))
        test_value = result.scalar()
        if test_value == 1:
            print("✅ Database connection test successful")
        else:
            print("❌ Database connection test failed")
            
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        print(f"   Database URI: {app.config.get('SQLALCHEMY_DATABASE_URI', 'Not set')}")
        # Don't crash the app, just log the error

@app.shell_context_processor
def make_shell_context():
    return {
        'db': db, 
        'User': User, 
        'School': School,
        'Program': Program,
        'Survey': Survey,
        'SurveyResponse': SurveyResponse,
        'Recommendation': Recommendation,

    }

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

