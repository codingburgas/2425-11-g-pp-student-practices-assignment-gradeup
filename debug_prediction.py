#!/usr/bin/env python3
"""
Debug script to find the array comparison issue
"""

from app import create_app, db
from app.models import User
from app.ml.prediction_system import advanced_prediction_system
from app.ml.demo_prediction_service import demo_prediction_service

def debug_prediction():
    """Debug the prediction system step by step"""
    print("🔍 Debugging Prediction System")
    print("=" * 50)
    
    app = create_app()
    
    with app.app_context():
        # Create test user
        test_user = User.query.filter_by(username='debug_user').first()
        if not test_user:
            test_user = User(
                username='debug_user',
                email='debug@example.com'
            )
            test_user.set_password('debug_password')
            db.session.add(test_user)
            db.session.commit()
        
        print("✅ Test user created/found")
        
        # Test survey data
        survey_data = {
            'career_interests': ['technology', 'data'],
            'favorite_subjects': ['mathematics', 'computer_science'],
            'career_goals': 'Data Scientist'
        }
        
        print("✅ Survey data prepared")
        
        # Test demo service directly
        print("\n📊 Testing demo service...")
        try:
            demo_predictions = demo_prediction_service.predict_programs(survey_data, top_k=5)
            print(f"✅ Demo service returned {len(demo_predictions)} predictions")
            
            for i, pred in enumerate(demo_predictions[:2], 1):
                print(f"   {i}. {pred['program_name']} - {pred['confidence']:.3f}")
        except Exception as e:
            print(f"❌ Demo service error: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test advanced prediction system step by step
        print("\n🧠 Testing advanced prediction system...")
        
        # Step 1: Test ML service check
        print("Step 1: Checking ML service...")
        try:
            is_trained = advanced_prediction_system.ml_service.is_trained
            print(f"   ML service trained: {is_trained}")
        except Exception as e:
            print(f"   ❌ ML service check error: {e}")
        
        # Step 2: Test feature extraction
        print("Step 2: Testing feature extraction...")
        try:
            from app.ml.utils import extract_features_from_survey_response
            features = extract_features_from_survey_response(survey_data)
            print(f"   ✅ Features extracted: {type(features)} with {len(features) if features else 0} items")
            if features:
                print(f"   Sample feature keys: {list(features.keys())[:3]}")
        except Exception as e:
            print(f"   ❌ Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 3: Test confidence enhancement (this is likely where the error is)
        print("Step 3: Testing confidence enhancement...")
        try:
            enhanced_predictions = advanced_prediction_system._enhance_predictions_with_confidence(
                demo_predictions, features, survey_data
            )
            print(f"   ✅ Enhanced {len(enhanced_predictions)} predictions")
        except Exception as e:
            print(f"   ❌ Confidence enhancement error: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to debug this specific step
            print("   🔍 Debugging confidence enhancement...")
            try:
                for pred in demo_predictions[:1]:  # Test with just one prediction
                    print(f"   Testing prediction: {pred}")
                    base_confidence = pred.get('confidence', 0.5)
                    print(f"   Base confidence: {base_confidence} (type: {type(base_confidence)})")
                    
                    # Test feature completeness
                    feature_completeness = advanced_prediction_system._calculate_feature_completeness(features)
                    print(f"   Feature completeness: {feature_completeness}")
                    
            except Exception as inner_e:
                print(f"   ❌ Inner debug error: {inner_e}")
                import traceback
                traceback.print_exc()
            
            return
        
        # Step 4: Test full prediction
        print("Step 4: Testing full prediction...")
        try:
            result = advanced_prediction_system.predict_with_confidence(
                survey_data=survey_data,
                user_id=test_user.id,
                top_k=3,
                store_history=False
            )
            print(f"   ✅ Full prediction successful: {len(result.get('predictions', []))} predictions")
        except Exception as e:
            print(f"   ❌ Full prediction error: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n🎉 All steps completed successfully!")

if __name__ == "__main__":
    debug_prediction() 