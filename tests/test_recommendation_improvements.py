"""
Test cases for the improved recommendation engine.
"""

import unittest
from unittest.mock import patch, MagicMock
from app.ml.recommendation_engine import RecommendationEngine
from app.models import School, Program, User

class TestRecommendationImprovements(unittest.TestCase):
    """Test cases for the recommendation engine improvements."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = RecommendationEngine()
        self.engine.is_initialized = True
        
        # Mock survey data
        self.survey_data = {
            'math_interest': 8,
            'science_interest': 6,
            'art_interest': 4,
            'sports_interest': 3,
            'career_goal': 'Technology',
            'grades_average': 5.5,
            'study_hours_per_day': 5
        }
        
        # Mock user preferences
        self.user_preferences = {
            'preferred_location': 'Sofia',
            'max_tuition': 5000
        }

    @patch('app.ml.recommendation_engine.School')
    def test_university_match_score(self, mock_school):
        """Test the improved university match score calculation."""
        # Create a mock school
        school = MagicMock()
        school.location = 'Sofia'
        school.name = 'Sofia University'
        school.description = 'A leading university in technology and science'
        school.programs.count.return_value = 15
        
        # Calculate match score
        score = self.engine._calculate_university_match_score(
            school, self.user_preferences, self.survey_data
        )
        
        # Assert score is within expected range
        self.assertGreater(score, 0.3)
        self.assertLessEqual(score, 1.0)

    @patch('app.ml.recommendation_engine.Program')
    def test_interest_alignment(self, mock_program):
        """Test the improved interest alignment calculation."""
        # Create a mock program
        program = MagicMock()
        program.name = 'Computer Science'
        program.description = 'A program focused on software development and algorithms'
        
        # Calculate alignment score
        score = self.engine._calculate_interest_alignment(program, self.survey_data)
        
        # Assert score is within expected range
        self.assertGreater(score, 0.1)
        self.assertLessEqual(score, 0.45)

    @patch('app.ml.recommendation_engine.Program')
    def test_career_alignment(self, mock_program):
        """Test the improved career alignment calculation."""
        # Create a mock program
        program = MagicMock()
        program.name = 'Software Engineering'
        program.description = 'A program focused on software development'
        
        # Calculate alignment score
        score = self.engine._calculate_career_alignment(program, self.survey_data)
        
        # Assert score is within expected range
        self.assertGreater(score, 0.1)
        self.assertLessEqual(score, 0.35)

    @patch('app.ml.recommendation_engine.School')
    def test_match_reasons(self, mock_school):
        """Test the improved match reasons generation."""
        # Create a mock school
        school = MagicMock()
        school.location = 'Sofia'
        school.name = 'Sofia University'
        school.programs.count.return_value = 15
        
        # Get match reasons
        reasons = self.engine._get_match_reasons(
            school, self.user_preferences, self.survey_data
        )
        
        # Assert reasons are generated
        self.assertGreater(len(reasons), 0)
        self.assertLessEqual(len(reasons), 3)

if __name__ == '__main__':
    unittest.main() 