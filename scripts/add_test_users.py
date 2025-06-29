#!/usr/bin/env python3
"""
Add 20 Test Users for System Testing

This script adds 20 diverse test users to test the recommendation system
with a larger user base.
"""

import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def add_test_users():
    """Add 20 test users to the database."""
    try:
        from app import create_app, db
        from app.models import User
        import json
        
        app = create_app()
        
        with app.app_context():
            print("👥 Adding 20 Test Users for System Testing")
            print("=" * 60)
            
            # Check existing users
            existing_users = User.query.count()
            print(f"📊 Current users in database: {existing_users}")
            
            # Sample user data with diverse profiles
            sample_users = [
                {
                    'username': 'alex_smith',
                    'email': 'alex.smith@example.com',
                    'password': 'password123',
                    'bio': 'Aspiring software engineer with a passion for AI and machine learning.',
                    'location': 'Sofia, Bulgaria'
                },
                {
                    'username': 'maria_popova',
                    'email': 'maria.popova@example.com', 
                    'password': 'password123',
                    'bio': 'Future doctor interested in pediatric medicine.',
                    'location': 'Plovdiv, Bulgaria'
                },
                {
                    'username': 'john_peterson',
                    'email': 'john.peterson@example.com',
                    'password': 'password123', 
                    'bio': 'Business student with entrepreneurial dreams.',
                    'location': 'Varna, Bulgaria'
                },
                {
                    'username': 'elena_dimitrova',
                    'email': 'elena.dimitrova@example.com',
                    'password': 'password123',
                    'bio': 'Art enthusiast pursuing graphic design studies.',
                    'location': 'Burgas, Bulgaria'
                },
                {
                    'username': 'david_johnson',
                    'email': 'david.johnson@example.com',
                    'password': 'password123',
                    'bio': 'Engineering student focused on renewable energy.',
                    'location': 'Sofia, Bulgaria'
                },
                {
                    'username': 'anna_petrova',
                    'email': 'anna.petrova@example.com',
                    'password': 'password123',
                    'bio': 'Psychology major interested in cognitive behavioral therapy.',
                    'location': 'Stara Zagora, Bulgaria'
                },
                {
                    'username': 'mike_brown',
                    'email': 'mike.brown@example.com',
                    'password': 'password123',
                    'bio': 'Mathematics student with a love for theoretical physics.',
                    'location': 'Plovdiv, Bulgaria'
                },
                {
                    'username': 'sofia_ivanova',
                    'email': 'sofia.ivanova@example.com',
                    'password': 'password123',
                    'bio': 'Literature student aspiring to become a writer.',
                    'location': 'Sofia, Bulgaria'
                },
                {
                    'username': 'robert_garcia',
                    'email': 'robert.garcia@example.com',
                    'password': 'password123',
                    'bio': 'International business student planning to work abroad.',
                    'location': 'Varna, Bulgaria'
                },
                {
                    'username': 'nina_kostova',
                    'email': 'nina.kostova@example.com',
                    'password': 'password123',
                    'bio': 'Biology student interested in marine conservation.',
                    'location': 'Burgas, Bulgaria'
                },
                {
                    'username': 'tom_anderson',
                    'email': 'tom.anderson@example.com',
                    'password': 'password123',
                    'bio': 'Computer science student specializing in cybersecurity.',
                    'location': 'Sofia, Bulgaria'
                },
                {
                    'username': 'svetlana_nikolova',
                    'email': 'svetlana.nikolova@example.com',
                    'password': 'password123',
                    'bio': 'Economics student with interest in financial markets.',
                    'location': 'Plovdiv, Bulgaria'
                },
                {
                    'username': 'james_wilson',
                    'email': 'james.wilson@example.com',
                    'password': 'password123',
                    'bio': 'Mechanical engineering student passionate about robotics.',
                    'location': 'Sofia, Bulgaria'
                },
                {
                    'username': 'diana_georgieva',
                    'email': 'diana.georgieva@example.com',
                    'password': 'password123',
                    'bio': 'Pharmacy student interested in drug development.',
                    'location': 'Sofia, Bulgaria'
                },
                {
                    'username': 'chris_taylor',
                    'email': 'chris.taylor@example.com',
                    'password': 'password123',
                    'bio': 'Fine arts student specializing in digital media.',
                    'location': 'Varna, Bulgaria'
                },
                {
                    'username': 'maria_stankova',
                    'email': 'maria.stankova@example.com',
                    'password': 'password123',
                    'bio': 'History student focused on medieval Bulgarian history.',
                    'location': 'Veliko Tarnovo, Bulgaria'
                },
                {
                    'username': 'daniel_lee',
                    'email': 'daniel.lee@example.com',
                    'password': 'password123',
                    'bio': 'Civil engineering student interested in smart cities.',
                    'location': 'Plovdiv, Bulgaria'
                },
                {
                    'username': 'violeta_todorova',
                    'email': 'violeta.todorova@example.com',
                    'password': 'password123',
                    'bio': 'Music performance student specializing in classical piano.',
                    'location': 'Sofia, Bulgaria'
                },
                {
                    'username': 'ryan_mitchell',
                    'email': 'ryan.mitchell@example.com',
                    'password': 'password123',
                    'bio': 'Finance student interested in cryptocurrency and blockchain.',
                    'location': 'Sofia, Bulgaria'
                },
                {
                    'username': 'petya_angelova',
                    'email': 'petya.angelova@example.com',
                    'password': 'password123',
                    'bio': 'Chemistry student researching sustainable materials.',
                    'location': 'Burgas, Bulgaria'
                }
            ]
            
            print(f"\n👤 Creating {len(sample_users)} test users...")
            created_count = 0
            skipped_count = 0
            
            for user_data in sample_users:
                # Check if user already exists
                existing_user = User.query.filter(
                    (User.username == user_data['username']) | 
                    (User.email == user_data['email'])
                ).first()
                
                if existing_user:
                    print(f"   ⚠️  User already exists: {user_data['username']}")
                    skipped_count += 1
                    continue
                
                # Create new user
                user = User(
                    username=user_data['username'],
                    email=user_data['email'],
                    bio=user_data.get('bio', ''),
                    location=user_data.get('location', ''),
                    email_verified=True,  # Pre-verify for testing
                    is_admin=False
                )
                
                # Set password using the model's method
                user.set_password(user_data['password'])
                
                # Add some random preferences for testing
                preferences = {
                    'study_time_preference': random.choice(['morning', 'afternoon', 'evening']),
                    'learning_style': random.choice(['visual', 'auditory', 'kinesthetic']),
                    'career_focus': random.choice(['research', 'industry', 'entrepreneurship', 'academia']),
                    'budget_range': random.choice(['low', 'medium', 'high']),
                    'location_preference': random.choice(['same_city', 'anywhere_in_country', 'international'])
                }
                user.set_preferences(preferences)
                
                db.session.add(user)
                print(f"   ✅ Created: {user_data['username']} ({user_data['email']})")
                created_count += 1
            
            # Commit all changes
            db.session.commit()
            
            print("\n" + "=" * 60)
            print("🎉 Test Users Creation Complete!")
            
            # Final statistics
            final_users = User.query.count()
            admin_users = User.query.filter_by(is_admin=True).count()
            regular_users = final_users - admin_users
            
            print(f"\n📊 User Statistics:")
            print(f"   - Total users: {final_users}")
            print(f"   - Admin users: {admin_users}")
            print(f"   - Regular users: {regular_users}")
            print(f"   - Users created this session: {created_count}")
            print(f"   - Users skipped (already exist): {skipped_count}")
            
            print(f"\n🔐 Test User Login Info:")
            print(f"   - All test users have password: 'password123'")
            print(f"   - All test users are email verified")
            print(f"   - Try logging in as: alex_smith, maria_popova, etc.")
            
            print(f"\n🚀 Ready to test with {final_users} users!")
            
    except Exception as e:
        print(f"❌ Error adding test users: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    add_test_users() 