
"""
ML Model Test Runner

This script provides a comprehensive testing suite for ML models with various options:
- Unit tests for core ML components
- Integration tests for the full ML pipeline
- Performance and benchmark tests
- Stress tests for scalability

Usage:
    python run_ml_tests.py --help
    python run_ml_tests.py --unit
    python run_ml_tests.py --performance
    python run_ml_tests.py --all
"""

import sys
import os
import argparse
import unittest
import time
from io import StringIO
import psutil

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MLTestResult(unittest.TextTestResult):
    """Custom test result class with enhanced reporting."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_times = {}
        self.start_time = None
        
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
        
    def stopTest(self, test):
        super().stopTest(test)
        if self.start_time:
            self.test_times[str(test)] = time.time() - self.start_time
            
    def printErrors(self):
        super().printErrors()
        if self.test_times:
            print("\n" + "="*70)
            print("TEST EXECUTION TIMES:")
            print("="*70)
            for test, duration in sorted(self.test_times.items(), key=lambda x: x[1], reverse=True):
                print(f"{test:<60} {duration:.3f}s")


class MLTestRunner(unittest.TextTestRunner):
    """Enhanced test runner with timing and system info."""
    
    resultclass = MLTestResult
    
    def run(self, test):
        print("="*70)
        print("ML MODEL TEST SUITE")
        print("="*70)
        
        # System information
        print(f"Python version: {sys.version}")
        print(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024:.0f} MB")
        print(f"CPU count: {psutil.cpu_count()}")
        print(f"Platform: {sys.platform}")
        print("="*70)
        
        start_time = time.time()
        result = super().run(test)
        end_time = time.time()
        
        print("="*70)
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        print("="*70)
        
        return result


def discover_tests(test_pattern):
    """Discover tests matching the given pattern."""
    loader = unittest.TestLoader()
    return loader.discover('tests', pattern=test_pattern)


def run_unit_tests():
    """Run unit tests for ML components."""
    print("Running ML Unit Tests...")
    suite = discover_tests('test_ml_models.py')
    runner = MLTestRunner(verbosity=2)
    return runner.run(suite)


def run_performance_tests():
    """Run performance and benchmark tests."""
    print("Running ML Performance Tests...")
    suite = discover_tests('test_ml_performance.py')
    runner = MLTestRunner(verbosity=2)
    return runner.run(suite)


def run_all_tests():
    """Run all ML tests."""
    print("Running All ML Tests...")
    suite = unittest.TestSuite()
    
    # Add all test modules
    suite.addTests(discover_tests('test_ml_models.py'))
    suite.addTests(discover_tests('test_ml_performance.py'))
    
    runner = MLTestRunner(verbosity=2)
    return runner.run(suite)


def run_specific_test(test_class, test_method=None):
    """Run a specific test class or method."""
    if test_method:
        suite = unittest.TestSuite()
        suite.addTest(test_class(test_method))
    else:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    
    runner = MLTestRunner(verbosity=2)
    return runner.run(suite)


def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        'numpy', 'psutil', 'unittest.mock'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"Error: Missing required modules: {', '.join(missing)}")
        print("Install them with: pip install " + " ".join(missing))
        return False
    
    return True


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="ML Model Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ml_tests.py --unit           # Run only unit tests
  python run_ml_tests.py --performance    # Run only performance tests
  python run_ml_tests.py --all            # Run all tests
  python run_ml_tests.py --fast           # Run fast tests only
  python run_ml_tests.py --benchmark      # Run benchmark comparisons
        """
    )
    
    parser.add_argument('--unit', action='store_true',
                       help='Run unit tests for ML models')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance and stress tests')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--fast', action='store_true',
                       help='Run only fast tests (skip slow performance tests)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark comparison tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output (errors only)')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Set verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # Determine what tests to run
    if args.unit:
        result = run_unit_tests()
    elif args.performance:
        result = run_performance_tests()
    elif args.all:
        result = run_all_tests()
    elif args.fast:
        # Run unit tests only (they're fast)
        result = run_unit_tests()
    elif args.benchmark:
        # Run only benchmark tests
        from tests.test_ml_performance import TestMLBenchmarks
        result = run_specific_test(TestMLBenchmarks)
    else:
        # Default: run unit tests
        print("No specific test type specified. Running unit tests by default.")
        print("Use --help to see all options.")
        result = run_unit_tests()
    
    # Return appropriate exit code
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 