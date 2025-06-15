import unittest
import json
from app import create_app, db
from app.models import User, School, Program, Survey

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        
        # Create test user with admin role
        self.admin = User(username='admin', email='admin@example.com', is_admin=True)
        self.admin.set_password('adminpass')
        
        # Create regular user
        self.user = User(username='user', email='user@example.com')
        self.user.set_password('userpass')
        
        # Create test data
        self.school = School(name='API Test School', location='Test Location')
        
        db.session.add_all([self.admin, self.user, self.school])
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def get_admin_token(self):
        response = self.client.post('/auth/login', data={
            'username': 'admin',
            'password': 'adminpass'
        })
        return response.cookies.get('session')

    def get_user_token(self):
        response = self.client.post('/auth/login', data={
            'username': 'user',
            'password': 'userpass'
        })
        return response.cookies.get('session')

    def test_get_schools_endpoint(self):
        # Test retrieving all schools
        response = self.client.get('/api/schools')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertGreaterEqual(len(data), 1)
        self.assertEqual(data[0]['name'], 'API Test School')
    
    def test_get_school_by_id(self):
        # Test retrieving a specific school
        response = self.client.get(f'/api/schools/{self.school.id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['name'], 'API Test School')
    
    def test_create_school_endpoint(self):
        # Login as admin
        with self.client as c:
            with c.session_transaction() as sess:
                sess['user_id'] = self.admin.id
                sess['is_admin'] = True
            
            # Test creating a new school
            response = c.post('/api/schools', json={
                'name': 'New API School',
                'location': 'New Location'
            })
            self.assertEqual(response.status_code, 201)
            
            # Verify the school was created
            school = School.query.filter_by(name='New API School').first()
            self.assertIsNotNone(school)
    
    def test_update_school_endpoint(self):
        # Login as admin
        with self.client as c:
            with c.session_transaction() as sess:
                sess['user_id'] = self.admin.id
                sess['is_admin'] = True
            
            # Test updating a school
            response = c.put(f'/api/schools/{self.school.id}', json={
                'name': 'Updated School',
                'location': 'Updated Location'
            })
            self.assertEqual(response.status_code, 200)
            
            # Verify the school was updated
            school = School.query.get(self.school.id)
            self.assertEqual(school.name, 'Updated School')
    
    def test_non_admin_cannot_modify_schools(self):
        # Login as regular user
        with self.client as c:
            with c.session_transaction() as sess:
                sess['user_id'] = self.user.id
                sess['is_admin'] = False
            
            # Try to create a new school
            response = c.post('/api/schools', json={
                'name': 'Unauthorized School',
                'location': 'Unauthorized Location'
            })
            self.assertIn(response.status_code, [401, 403])
            
            # Verify the school was not created
            school = School.query.filter_by(name='Unauthorized School').first()
            self.assertIsNone(school)

if __name__ == '__main__':
    unittest.main() 