import unittest
from app import create_app, db
from app.models import User, School, Program, Survey, SurveyResponse, Recommendation
from datetime import datetime
import json

class ModelsTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_user_password_hashing(self):
        u = User(username='test_user', email='test@example.com')
        u.set_password('password123')
        self.assertFalse(u.check_password('wrong_password'))
        self.assertTrue(u.check_password('password123'))

    def test_user_relationships(self):
        # Create user
        user = User(username='student', email='student@example.com')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        
        # Create a school
        school = School(name='Test University', location='Test City')
        db.session.add(school)
        
        # Create a program associated with the school
        program = Program(
            name='Computer Science',
            description='CS program',
            school=school,
            degree_type='Bachelor'  # Adding required field
        )
        db.session.add(program)
        
        # Create a survey with questions
        questions = [
            {"id": 1, "text": "What are your interests?", "type": "text"}
        ]
        survey = Survey(
            title='Career Survey',
            questions=json.dumps(questions)
        )
        db.session.add(survey)
        db.session.commit()
        
        # Create a survey response linked to user and survey
        response = SurveyResponse(
            user_id=user.id,
            survey_id=survey.id,
            answers=json.dumps({"1": "Programming"})
        )
        db.session.add(response)
        db.session.commit()
        
        # Refresh user object to ensure relationships are loaded
        db.session.refresh(user)
        
        # Check relationships
        self.assertEqual(response.user_id, user.id)
        self.assertEqual(program.school_id, school.id)
        self.assertIn(response, user.survey_responses.all())

    def test_recommendation_creation(self):
        # Create user
        user = User(username='rec_test', email='rec@example.com')
        db.session.add(user)
        db.session.commit()
        
        # Create school and program
        school = School(name='Rec University', location='Rec City')
        db.session.add(school)
        db.session.commit()
        
        program = Program(
            name='Data Science',
            description='Data program',
            school=school,
            degree_type='Master'  # Adding required field
        )
        db.session.add(program)
        db.session.commit()
        
        # Create survey and response
        survey = Survey(
            title='Career Survey',
            questions=json.dumps([{"id": 1, "text": "Skills?", "type": "text"}])
        )
        db.session.add(survey)
        db.session.commit()
        
        response = SurveyResponse(
            user_id=user.id,
            survey_id=survey.id,
            answers=json.dumps({"1": "Python, Data Analysis"})
        )
        db.session.add(response)
        db.session.commit()
        
        # Create recommendation
        recommendation = Recommendation(
            survey_response_id=response.id,
            program_id=program.id,
            score=85.5
        )
        
        db.session.add(recommendation)
        db.session.commit()
        
        # Check recommendation
        self.assertEqual(recommendation.survey_response_id, response.id)
        self.assertEqual(recommendation.program_id, program.id)
        self.assertEqual(recommendation.score, 85.5)

if __name__ == '__main__':
    unittest.main() 