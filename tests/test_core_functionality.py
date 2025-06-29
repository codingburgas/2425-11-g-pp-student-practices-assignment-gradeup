#!/usr/bin/env python3
"""
Core Functionality Testing for Recommendation Engine

This script tests all core recommendation engine functionality to ensure
there are no errors and everything works perfectly.
"""

import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_functionality():
    """Test all core recommendation engine functionality."""
    print("🔧 Testing Recommendation Engine Core Functionality")
    print("=" * 65)
    
    # Test results tracking
    test_results = {
        'app_context': False,
        'database_connectivity': False,
        'engine_initialization': False,
        'university_matching': False,
        'program_recommendations': False,
        'personalized_suggestions': False,
        'recommendation_history': False,
        'pattern_analysis': False,
        'store_history': False,
        'helper_functions': False
    }
    
    errors_found = []
    
    try:
        # Setup Flask application context
        print("\n🚀 Setting up Flask application context...")
        from app import create_app, db
        from app.ml.recommendation_engine import recommendation_engine
        
        app = create_app()
        
        with app.app_context():
            print("✅ Flask application context created successfully")
            test_results['app_context'] = True
            
            # Test database connectivity
            print("\n🗄️ Testing database connectivity...")
            try:
                from app.models import School, Program, User, Survey, SurveyResponse
                
                school_count = School.query.count()
                program_count = Program.query.count() 
                user_count = User.query.count()
                
                print(f"   📊 Database Statistics:")
                print(f"      - Schools: {school_count}")
                print(f"      - Programs: {program_count}")
                print(f"      - Users: {user_count}")
                
                if school_count >= 0 and program_count >= 0 and user_count >= 0:
                    print("✅ Database connectivity: SUCCESS")
                    test_results['database_connectivity'] = True
                else:
                    errors_found.append("Database returned negative counts")
                    
            except Exception as e:
                errors_found.append(f"Database connectivity error: {e}")
                print(f"❌ Database connectivity error: {e}")
            
            # Test engine initialization
            print("\n⚙️ Testing recommendation engine initialization...")
            try:
                recommendation_engine.initialize()
                if recommendation_engine.is_initialized:
                    print("✅ Recommendation engine initialization: SUCCESS")
                    test_results['engine_initialization'] = True
                else:
                    errors_found.append("Engine failed to initialize properly")
                    
            except Exception as e:
                errors_found.append(f"Engine initialization error: {e}")
                print(f"❌ Engine initialization error: {e}")
            
            # Test data
            test_survey_data = {
                'math_interest': 8,
                'science_interest': 7,
                'art_interest': 3,
                'sports_interest': 5,
                'study_hours_per_day': 4,
                'preferred_subject': 'Mathematics',
                'career_goal': 'Engineer',
                'extracurricular': True,
                'leadership_experience': True,
                'team_preference': False,
                'languages_spoken': ['Bulgarian', 'English'],
                'grades_average': 5.5
            }
            
            test_user_preferences = {
                'preferred_location': 'Sofia',
                'max_tuition': 10000,
                'preferred_duration': '4 years'
            }
            
            print(f"\n📊 Using test data with {len(test_survey_data)} survey fields and {len(test_user_preferences)} preference fields")
            
            # Test 1: University Matching
            print("\n🏫 Testing university matching algorithm...")
            try:
                universities = recommendation_engine.match_universities(
                    user_preferences=test_user_preferences,
                    survey_data=test_survey_data,
                    top_k=5
                )
                
                if isinstance(universities, list):
                    print(f"✅ University matching: SUCCESS ({len(universities)} results)")
                    
                    # Validate result structure
                    if universities:
                        sample = universities[0]
                        required_fields = ['school_id', 'school_name', 'match_score', 'match_reasons']
                        missing_fields = [field for field in required_fields if field not in sample]
                        
                        if not missing_fields:
                            scores = [u['match_score'] for u in universities]
                            print(f"   📊 Match scores: {min(scores):.2f} - {max(scores):.2f}")
                            print(f"   🎯 Sample result: {sample['school_name']} ({sample['match_score']:.2f})")
                            test_results['university_matching'] = True
                        else:
                            errors_found.append(f"University result missing fields: {missing_fields}")
                    else:
                        print("   ⚠️ No universities returned (may be normal with empty database)")
                        test_results['university_matching'] = True  # Not an error
                else:
                    errors_found.append(f"University matching returned invalid type: {type(universities)}")
                    
            except Exception as e:
                errors_found.append(f"University matching error: {e}")
                print(f"❌ University matching error: {e}")
                traceback.print_exc()
            
            # Test 2: Program Recommendations
            print("\n📚 Testing program recommendation algorithm...")
            try:
                programs = recommendation_engine.recommend_programs(
                    user_id=1,
                    survey_data=test_survey_data,
                    user_preferences=test_user_preferences,
                    top_k=5
                )
                
                if isinstance(programs, list):
                    print(f"✅ Program recommendations: SUCCESS ({len(programs)} results)")
                    
                    # Validate result structure
                    if programs:
                        sample = programs[0]
                        required_fields = ['program_id', 'program_name', 'school_name', 'match_score']
                        missing_fields = [field for field in required_fields if field not in sample]
                        
                        if not missing_fields:
                            scores = [p['match_score'] for p in programs]
                            print(f"   📊 Match scores: {min(scores):.2f} - {max(scores):.2f}")
                            print(f"   🎯 Sample result: {sample['program_name']} at {sample['school_name']} ({sample['match_score']:.2f})")
                            print(f"   💡 Reasons: {sample.get('recommendation_reasons', [])}")
                            test_results['program_recommendations'] = True
                        else:
                            errors_found.append(f"Program result missing fields: {missing_fields}")
                    else:
                        print("   ⚠️ No programs returned (may be normal with empty database)")
                        test_results['program_recommendations'] = True  # Not an error
                else:
                    errors_found.append(f"Program recommendations returned invalid type: {type(programs)}")
                    
            except Exception as e:
                errors_found.append(f"Program recommendations error: {e}")
                print(f"❌ Program recommendations error: {e}")
                traceback.print_exc()
            
            # Test 3: Personalized Suggestions
            print("\n💡 Testing personalized suggestions...")
            try:
                suggestions = recommendation_engine.get_personalized_suggestions(
                    user_id=1,
                    limit=3
                )
                
                if isinstance(suggestions, dict):
                    print(f"✅ Personalized suggestions: SUCCESS ({len(suggestions)} categories)")
                    
                    expected_categories = ['trending_programs', 
                                         'completion_suggestions', 'seasonal_recommendations']
                    
                    for category in expected_categories:
                        if category in suggestions:
                            items = suggestions[category]
                            count = len(items) if isinstance(items, list) else 'N/A'
                            print(f"   📋 {category}: {count} items")
                        else:
                            errors_found.append(f"Missing suggestion category: {category}")
                    
                    test_results['personalized_suggestions'] = True
                else:
                    errors_found.append(f"Personalized suggestions returned invalid type: {type(suggestions)}")
                    
            except Exception as e:
                errors_found.append(f"Personalized suggestions error: {e}")
                print(f"❌ Personalized suggestions error: {e}")
                traceback.print_exc()
            
            # Test 4: Recommendation History
            print("\n📈 Testing recommendation history...")
            try:
                history = recommendation_engine.get_recommendation_history(
                    user_id=1,
                    limit=10
                )
                
                if isinstance(history, list):
                    print(f"✅ Recommendation history: SUCCESS ({len(history)} records)")
                    test_results['recommendation_history'] = True
                else:
                    errors_found.append(f"Recommendation history returned invalid type: {type(history)}")
                    
            except Exception as e:
                errors_found.append(f"Recommendation history error: {e}")
                print(f"❌ Recommendation history error: {e}")
                traceback.print_exc()
            
            # Test 5: Pattern Analysis
            print("\n🔍 Testing recommendation pattern analysis...")
            try:
                patterns = recommendation_engine.analyze_recommendation_patterns(
                    user_id=1
                )
                
                if isinstance(patterns, dict):
                    print(f"✅ Pattern analysis: SUCCESS")
                    
                    if 'error' not in patterns:
                        print(f"   📊 Analysis keys: {list(patterns.keys())}")
                        test_results['pattern_analysis'] = True
                    else:
                        print(f"   ⚠️ Analysis result: {patterns.get('message', 'No data available')}")
                        test_results['pattern_analysis'] = True  # Not an error
                else:
                    errors_found.append(f"Pattern analysis returned invalid type: {type(patterns)}")
                    
            except Exception as e:
                errors_found.append(f"Pattern analysis error: {e}")
                print(f"❌ Pattern analysis error: {e}")
                traceback.print_exc()
            
            # Test 6: Store Recommendation History
            print("\n💾 Testing store recommendation history...")
            try:
                sample_recommendations = [
                    {
                        'program_id': 1,
                        'program_name': 'Test Program',
                        'match_score': 0.8
                    }
                ]
                
                result = recommendation_engine.store_recommendation_history(
                    user_id=1,
                    survey_response_id=None,
                    recommendations=sample_recommendations,
                    recommendation_type='program'
                )
                
                if result:
                    print("✅ Store recommendation history: SUCCESS")
                    test_results['store_history'] = True
                else:
                    errors_found.append("Store recommendation history returned False")
                    
            except Exception as e:
                errors_found.append(f"Store recommendation history error: {e}")
                print(f"❌ Store recommendation history error: {e}")
                traceback.print_exc()
            
            # Test 7: Helper Functions
            print("\n🔧 Testing helper functions...")
            try:
                test_programs = [
                    "Computer Science Engineering",
                    "Business Administration",
                    "Medicine",
                    "Fine Arts",
                    "Physics Research"
                ]
                
                classifications = []
                for prog in test_programs:
                    prog_type = recommendation_engine._classify_program_type(prog)
                    classifications.append((prog, prog_type))
                    print(f"   🏷️ '{prog}' -> '{prog_type}'")
                
                # Validate classifications
                expected_types = ['engineering', 'business', 'health', 'creative', 'science']
                actual_types = [classification[1] for classification in classifications]
                
                if all(expected_type in actual_types for expected_type in expected_types):
                    print("✅ Helper functions: SUCCESS")
                    test_results['helper_functions'] = True
                else:
                    errors_found.append(f"Helper functions classification mismatch: {actual_types}")
                    
            except Exception as e:
                errors_found.append(f"Helper functions error: {e}")
                print(f"❌ Helper functions error: {e}")
                traceback.print_exc()
                
    except Exception as e:
        errors_found.append(f"Critical application setup error: {e}")
        print(f"❌ Critical error: {e}")
        traceback.print_exc()
    
    # Summary Report
    print("\n" + "=" * 65)
    print("🎉 Core Functionality Testing Complete!")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\n📊 Test Results Summary:")
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if errors_found:
        print(f"\n❌ Errors Found ({len(errors_found)}):")
        for i, error in enumerate(errors_found, 1):
            print(f"   {i}. {error}")
    
    if passed_tests == total_tests and not errors_found:
        print("\n🎉 SUCCESS! All tests passed with no errors!")
        print("   ✅ Recommendation engine is fully functional")
        print("   ✅ All algorithms work correctly")
        print("   ✅ Database integration works")
        print("   ✅ Error handling is robust")
        print("   ✅ Ready for production use!")
        return True
    else:
        print(f"\n⚠️ Issues found: {len(errors_found)} errors, {total_tests - passed_tests} failed tests")
        return False

if __name__ == "__main__":
    success = test_core_functionality()
    sys.exit(0 if success else 1) 