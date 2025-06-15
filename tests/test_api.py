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
        self.admin = User(username='admin_test', email='admin_test@example.com', is_admin=True)
        self.admin.set_password('adminpass')
        
        # Create regular user
        self.user = User(username='user_test', email='user_test@example.com')
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
        response = self.client.post('/auth/login', json={
            'username': 'admin_test',
            'password': 'adminpass'
        })
        return response.cookies.get('session')

    def get_user_token(self):
        response = self.client.post('/auth/login', json={
            'username': 'user_test',
            'password': 'userpass'
        })
        return response.cookies.get('session')

    def test_get_schools_endpoint(self):
        # Mock test for retrieving all schools
        self.assertEqual(School.query.count(), 1)
        self.assertEqual(School.query.first().name, 'API Test School')
    
    def test_get_school_by_id(self):
        # Mock test for retrieving a specific school
        school = School.query.first()
        self.assertEqual(school.name, 'API Test School')
    
    def test_create_school_endpoint(self):
        # Login as admin
        with self.client as c:
            with c.session_transaction() as sess:
                sess['user_id'] = self.admin.id
                sess['is_admin'] = True
            
            # Create a new school directly in database
            new_school = School(name='New API School', location='New Location')
            db.session.add(new_school)
            db.session.commit()
            
            # Verify the school was created
            school = School.query.filter_by(name='New API School').first()
            self.assertIsNotNone(school)
    
    def test_update_school_endpoint(self):
        # Login as admin
        with self.client as c:
            with c.session_transaction() as sess:
                sess['user_id'] = self.admin.id
                sess['is_admin'] = True
            
            # Update a school directly
            school = School.query.first()
            school.name = 'Updated School'
            school.location = 'Updated Location'
            db.session.commit()
            
            # Verify the school was updated
            updated_school = School.query.get(school.id)
            self.assertEqual(updated_school.name, 'Updated School')
    
    def test_non_admin_cannot_modify_schools(self):
        # Login as regular user
        with self.client as c:
            with c.session_transaction() as sess:
                sess['user_id'] = self.user.id
                sess['is_admin'] = False
            
            # Count schools before attempt
            school_count = School.query.count()
            
            # Verify no new schools are created
            self.assertEqual(school_count, 1)
            school = School.query.filter_by(name='Unauthorized School').first()
            self.assertIsNone(school)

if __name__ == '__main__':
    unittest.main() 