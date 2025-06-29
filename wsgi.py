from app import create_app, db
from app.models import User, School, Program, Survey, SurveyResponse, Recommendation
import os

app = create_app()

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

