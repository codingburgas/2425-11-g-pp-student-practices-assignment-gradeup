#!/usr/bin/env python3
"""
Quick verification script for the prediction system
"""

from app import create_app, db
from sqlalchemy import inspect
from app.models import PredictionHistory, User

def check_database():
    """Check database components"""
    print("🗄️  Checking Database...")
    
    app = create_app()
    with app.app_context():
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        print(f"   📋 Total tables: {len(tables)}")
        
        if 'prediction_history' in tables:
            print("   ✅ PredictionHistory table exists")
            
            columns = inspector.get_columns('prediction_history')
            column_names = [col['name'] for col in columns]
            print(f"   📊 Columns: {column_names}")
            
            # Check if we can create a PredictionHistory object
            try:
                test_record = PredictionHistory()
                print("   ✅ PredictionHistory model can be instantiated")
            except Exception as e:
                print(f"   ❌ Error creating PredictionHistory: {e}")
        else:
            print("   ❌ PredictionHistory table not found")

def check_api_endpoints():
    """Check API endpoints"""
    print("\n🌐 Checking API Endpoints...")
    
    app = create_app()
    
    # Get all registered routes
    routes = []
    for rule in app.url_map.iter_rules():
        if 'prediction' in rule.rule:
            routes.append(f"{rule.methods} {rule.rule}")
    
    print(f"   📋 Found {len(routes)} prediction endpoints:")
    for route in routes:
        print(f"      {route}")

def check_prediction_system():
    """Check prediction system initialization"""
    print("\n🧠 Checking Prediction System...")
    
    try:
        from app.ml.prediction_system import advanced_prediction_system
        print("   ✅ Prediction system imported successfully")
        print(f"   📊 Model version: {advanced_prediction_system.model_version}")
        print(f"   🎯 Confidence threshold: {advanced_prediction_system.confidence_threshold}")
        
        # Test basic functionality
        try:
            test_data = {'career_interests': ['test']}
            result = advanced_prediction_system.predict_with_confidence(
                survey_data=test_data,
                user_id=1,
                store_history=False
            )
            print("   ✅ Basic prediction call successful")
            print(f"   📈 Result structure: {list(result.keys())}")
        except Exception as e:
            print(f"   ⚠️  Prediction call failed (expected without trained model): {str(e)[:100]}...")
            
    except Exception as e:
        print(f"   ❌ Failed to import prediction system: {e}")

def main():
    print("🎓 GradeUp Prediction System - Quick Check")
    print("=" * 50)
    
    try:
        check_database()
        check_api_endpoints()
        check_prediction_system()
        
        print("\n" + "=" * 50)
        print("✅ Quick check completed!")
        print("\n🚀 To test the full system:")
        print("   1. Run: python test_prediction_system.py")
        print("   2. Start Flask app: flask run")
        print("   3. Test API endpoints with curl or Postman")
        
    except Exception as e:
        print(f"\n❌ Check failed: {e}")

if __name__ == "__main__":
    main() 