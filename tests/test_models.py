import unittest
from app import create_app, db
from app.models import User, School, Program, Survey, SurveyResponse, Recommendation
from datetime import datetime

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
        
        # Create a school
        school = School(name='Test University', location='Test City')
        db.session.add(school)
        
        # Create a program associated with the school
        program = Program(
            name='Computer Science',
            description='CS program',
            school=school
        )
        db.session.add(program)
        
        # Create a survey for the user
        survey = Survey(title='Career Survey', user=user)
        db.session.add(survey)
        db.session.commit()
        
        # Check relationships
        self.assertEqual(survey.user_id, user.id)
        self.assertEqual(program.school_id, school.id)
        self.assertEqual(len(user.surveys.all()), 1)

    def test_recommendation_creation(self):
        # Create user and program
        user = User(username='rec_test', email='rec@example.com')
        school = School(name='Rec University', location='Rec City')
        program = Program(
            name='Data Science',
            description='Data program',
            school=school
        )
        
        # Create recommendation
        recommendation = Recommendation(
            user=user,
            program=program,
            score=85.5,
            match_reason="Strong analytical skills match"
        )
        
        db.session.add_all([user, school, program, recommendation])
        db.session.commit()
        
        # Check recommendation
        self.assertEqual(recommendation.user_id, user.id)
        self.assertEqual(recommendation.program_id, program.id)
        self.assertEqual(recommendation.score, 85.5)

if __name__ == '__main__':
    unittest.main() 