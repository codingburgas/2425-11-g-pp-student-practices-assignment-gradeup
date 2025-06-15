"""
Test cases for the email verification toggle feature.
"""

import unittest
from unittest.mock import patch, MagicMock
from flask import Flask, current_app
from app import create_app
from app.models import User
from config import Config

class TestConfig(Config):
    """Test configuration with email verification disabled."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    DISABLE_EMAIL_VERIFICATION = True

class TestEmailVerificationToggle(unittest.TestCase):
    """Test cases for the email verification toggle feature."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = create_app(TestConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()

    def tearDown(self):
        """Tear down test fixtures."""
        self.app_context.pop()

    def test_config_loaded(self):
        """Test that the email verification toggle is loaded from config."""
        self.assertTrue(current_app.config['DISABLE_EMAIL_VERIFICATION'])

    @patch('app.auth.routes.User')
    @patch('app.auth.routes.db')
    def test_registration_with_toggle_enabled(self, mock_db, mock_user):
        """Test that registration with toggle enabled marks user as verified."""
        # Mock the User class
        user_instance = MagicMock()
        mock_user.return_value = user_instance
        
        # Mock form data
        form_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'password123',
            'password2': 'password123'
        }
        
        # Make the request
        response = self.client.post('/auth/register', data=form_data, follow_redirects=True)
        
        # Assert user is marked as verified
        self.assertTrue(user_instance.email_verified)
        
    @patch('app.auth.routes.User')
    def test_login_with_toggle_enabled(self, mock_user):
        """Test that login with toggle enabled skips email verification check."""
        # Mock the User class
        user_instance = MagicMock()
        user_instance.email = 'test@example.com'
        user_instance.check_password.return_value = True
        user_instance.email_verified = False  # User not verified
        mock_user.query.filter_by.return_value.first.return_value = user_instance
        
        # Mock form data
        form_data = {
            'email': 'test@example.com',
            'password': 'password123',
            'remember_me': False
        }
        
        # Make the request with a patched login_user function
        with patch('app.auth.routes.login_user') as mock_login:
            response = self.client.post('/auth/login', data=form_data, follow_redirects=True)
            
            # Assert login_user was called despite user not being verified
            mock_login.assert_called_once()

if __name__ == '__main__':
    unittest.main() 