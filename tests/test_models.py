import unittest
from app import create_app, db
from app.models import User, School, Program, Survey, SurveyResponse, Recommendation, Favorite
from config import Config
import json

class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class ModelsTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
    
    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
    
    def test_user_model(self):
        u = User(username='testuser', email='test@example.com')
        u.set_password('password')
        db.session.add(u)
        db.session.commit()
        
        self.assertIsNotNone(User.query.filter_by(username='testuser').first())
        self.assertTrue(u.check_password('password'))
        self.assertFalse(u.check_password('wrongpassword'))
    
    def test_school_model(self):
        s = School(name='Test School', location='Test Location')
        db.session.add(s)
        db.session.commit()
        
        self.assertIsNotNone(School.query.filter_by(name='Test School').first())
    
    def test_program_model(self):
        s = School(name='Test School', location='Test Location')
        db.session.add(s)
        db.session.commit()
        
        p = Program(name='Test Program', degree_type='Bachelor', school_id=s.id)
        db.session.add(p)
        db.session.commit()
        
        self.assertIsNotNone(Program.query.filter_by(name='Test Program').first())
        self.assertEqual(p.school.name, 'Test School')
    
    def test_survey_model(self):
        questions = [
            {"id": 1, "text": "What are your favorite subjects?", "type": "multiple_choice", 
             "options": ["Math", "Science", "Art", "History"]},
            {"id": 2, "text": "How important is location?", "type": "rating", "min": 1, "max": 5}
        ]
        
        s = Survey(title='Academic Preferences', questions=json.dumps(questions))
        db.session.add(s)
        db.session.commit()
        
        survey = Survey.query.filter_by(title='Academic Preferences').first()
        self.assertIsNotNone(survey)
        self.assertEqual(len(survey.get_questions()), 2)
    
    def test_survey_response_model(self):
        
        u = User(username='testuser', email='test@example.com')
        db.session.add(u)
        
        
        questions = [{"id": 1, "text": "Test question?", "type": "text"}]
        s = Survey(title='Test Survey', questions=json.dumps(questions))
        db.session.add(s)
        db.session.commit()
        
        
        answers = {"1": "My answer"}
        sr = SurveyResponse(user_id=u.id, survey_id=s.id, answers=json.dumps(answers))
        db.session.add(sr)
        db.session.commit()
        
        response = SurveyResponse.query.filter_by(user_id=u.id, survey_id=s.id).first()
        self.assertIsNotNone(response)
        self.assertEqual(response.get_answers()["1"], "My answer")
    
    def test_recommendation_model(self):
        
        u = User(username='testuser', email='test@example.com')
        db.session.add(u)
        
        
        s = School(name='Test School', location='Test Location')
        db.session.add(s)
        db.session.commit()
        
        p = Program(name='Test Program', degree_type='Bachelor', school_id=s.id)
        db.session.add(p)
        db.session.commit()
        
        
        questions = [{"id": 1, "text": "Test question?", "type": "text"}]
        survey = Survey(title='Test Survey', questions=json.dumps(questions))
        db.session.add(survey)
        db.session.commit()
        
        answers = {"1": "My answer"}
        sr = SurveyResponse(user_id=u.id, survey_id=survey.id, answers=json.dumps(answers))
        db.session.add(sr)
        db.session.commit()
        
        
        r = Recommendation(survey_response_id=sr.id, program_id=p.id, score=0.85)
        db.session.add(r)
        db.session.commit()
        
        rec = Recommendation.query.filter_by(survey_response_id=sr.id, program_id=p.id).first()
        self.assertIsNotNone(rec)
        self.assertEqual(rec.score, 0.85)
    
    def test_favorite_model(self):
        
        u = User(username='testuser', email='test@example.com')
        db.session.add(u)
        
        
        s = School(name='Test School', location='Test Location')
        db.session.add(s)
        db.session.commit()
        
        
        f = Favorite(user_id=u.id, school_id=s.id)
        db.session.add(f)
        db.session.commit()
        
        fav = Favorite.query.filter_by(user_id=u.id, school_id=s.id).first()
        self.assertIsNotNone(fav)

if __name__ == '__main__':
    unittest.main() 