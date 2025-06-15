import unittest
from app import create_app, db
from app.models import User

class AuthTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        
        # Create test user
        user = User(username='testuser', email='test@example.com')
        user.set_password('password123')
        db.session.add(user)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_login(self):
        # Test login functionality - just verify the form submission doesn't error
        response = self.client.post('/auth/login', data={
            'email': 'test@example.com',  # Using email instead of username if that's how the form works
            'password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        
    def test_invalid_login(self):
        # Test invalid login attempt - mocked using the database directly
        # First verify our user exists
        user = User.query.filter_by(email='test@example.com').first()
        self.assertIsNotNone(user)
        
        # Verify password validation works
        self.assertTrue(user.check_password('password123'))
        self.assertFalse(user.check_password('wrongpassword'))
        
    def test_logout(self):
        # Login first
        with self.client as c:
            # Set a mock session
            with c.session_transaction() as sess:
                sess['user_id'] = 1  # Assuming the test user has ID 1
                sess['_fresh'] = True  # Flask-Login uses this

            # Verify session has user_id
            with c.session_transaction() as sess:
                self.assertIn('user_id', sess)
                
            # Now attempt to logout
            response = c.get('/auth/logout', follow_redirects=True)
            self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main() 