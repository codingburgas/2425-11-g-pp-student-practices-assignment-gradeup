import unittest
from unittest.mock import patch, MagicMock
from app import create_app, db
from app.models import User, School, Program, Survey, SurveyResponse, Recommendation

class MLRecommendationTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        
        # Create test user and login
        self.user = User(username='mltest', email='ml@example.com')
        self.user.set_password('password')
        db.session.add(self.user)
        
        # Create test school and programs
        self.school = School(name='ML University', location='ML City')
        db.session.add(self.school)
        
        self.program1 = Program(
            name='Data Science',
            description='Data Science program',
            school=self.school
        )
        
        self.program2 = Program(
            name='Machine Learning',
            description='ML program',
            school=self.school
        )
        
        db.session.add_all([self.program1, self.program2])
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    @patch('app.ml.recommendation_engine.recommendation_engine.generate_recommendations')
    def test_recommendation_generation(self, mock_generate):
        # Mock the recommendation engine response
        mock_recommendations = [
            {'program_id': self.program1.id, 'score': 95.0, 'match_reason': 'Strong match'},
            {'program_id': self.program2.id, 'score': 85.0, 'match_reason': 'Good match'}
        ]
        mock_generate.return_value = mock_recommendations
        
        # Create a survey and response
        survey = Survey(
            title='Career Survey',
            user=self.user
        )
        db.session.add(survey)
        
        response_data = {
            'interests': ['programming', 'data'],
            'skills': ['python', 'statistics'],
            'location_preference': 'urban'
        }
        
        survey_response = SurveyResponse(
            user=self.user,
            survey=survey,
            answers=str(response_data)
        )
        db.session.add(survey_response)
        db.session.commit()
        
        # Login as the user
        with self.client as c:
            with c.session_transaction() as sess:
                sess['user_id'] = self.user.id
            
            # Call the recommendation endpoint
            response = c.post(f'/api/recommendations/generate/{survey_response.id}', 
                             follow_redirects=True)
            
            # Verify the recommendation engine was called
            mock_generate.assert_called_once()
            
            # Check that recommendations were saved to the database
            recs = Recommendation.query.filter_by(user_id=self.user.id).all()
            self.assertEqual(len(recs), 2)  # Two recommendations should be created

    def test_recommendation_display(self):
        # Create recommendation directly in the database
        recommendation = Recommendation(
            user=self.user,
            program=self.program1,
            score=90.0,
            match_reason="Test reason"
        )
        db.session.add(recommendation)
        db.session.commit()
        
        # Login as the user
        with self.client as c:
            with c.session_transaction() as sess:
                sess['user_id'] = self.user.id
            
            # Request the recommendations page
            response = c.get('/recommendations', follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            
            # Check that the program appears in the response
            self.assertIn(b'ML University', response.data)
            self.assertIn(b'Data Science', response.data)

if __name__ == '__main__':
    unittest.main() 