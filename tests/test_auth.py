import unittest
from app import create_app, db
from app.models import User
from flask_login import current_user

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
        response = self.client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'password123',
            'remember_me': False
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        
    def test_invalid_login(self):
        response = self.client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'wrongpassword',
            'remember_me': False
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid username or password', response.data)
        
    def test_logout(self):
        # Login first
        self.client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'password123',
            'remember_me': False
        })
        # Then logout
        response = self.client.get('/auth/logout', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main() 