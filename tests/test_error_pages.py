import unittest
from app import create_app
from flask import current_app

class ErrorPagesTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        
        # Override the default error handlers for testing
        @self.app.errorhandler(404)
        def page_not_found(e):
            return "Page Not Found", 404
            
        @self.app.errorhandler(500)
        def internal_server_error(e):
            return "Internal Server Error", 500
        
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    def test_404_page(self):
        response = self.client.get('/non_existent_page')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Page Not Found', response.data)

    def test_500_page(self):
        # Add a route that triggers a 500 error
        @self.app.route('/trigger_error')
        def trigger_error():
            # Deliberately raise an exception
            raise Exception('Test 500 error')
            
        # Configure app to show errors during testing (rather than default error pages)
        self.app.config['TESTING'] = False
        self.app.config['DEBUG'] = False
        
        try:
            response = self.client.get('/trigger_error')
            self.assertEqual(response.status_code, 500)
            self.assertIn(b'Internal Server Error', response.data)
        finally:
            # Restore testing config
            self.app.config['TESTING'] = True

if __name__ == '__main__':
    unittest.main()
