import unittest
import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Discover and run all tests
def run_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    print("\n===============================================")
    print("RUNNING TESTS")
    print("===============================================\n")
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    print("\n===============================================")
    print(f"TEST RESULTS: {result.testsRun} tests run")
    print(f"FAILURES: {len(result.failures)}")
    print(f"ERRORS: {len(result.errors)}")
    print("===============================================\n")
    
    return result

if __name__ == '__main__':
    result = run_tests()
    if len(result.failures) > 0 or len(result.errors) > 0:
        sys.exit(1)
    sys.exit(0) 