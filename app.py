from app import create_app, db
from app.models import User, School, Program, Survey, SurveyResponse, Recommendation, Favorite

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
        'Favorite': Favorite
    }

if __name__ == '__main__':
    app.run(debug=True)