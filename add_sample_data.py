#!/usr/bin/env python3
"""
Add Sample Data for Recommendation Engine Testing

This script adds sample schools, programs, and users to test the recommendation engine.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def add_sample_data():
    """Add sample data to the database."""
    try:
        from app import create_app, db
        from app.models import School, Program, User, Survey, SurveyResponse, Favorite
        
        app = create_app()
        
        with app.app_context():
            print("🗄️ Adding Sample Data for Recommendation Engine Testing")
            print("=" * 60)
            
            # Check if data already exists
            existing_schools = School.query.count()
            existing_programs = Program.query.count()
            
            if existing_schools > 0 or existing_programs > 0:
                print(f"📊 Existing data found:")
                print(f"   - Schools: {existing_schools}")
                print(f"   - Programs: {existing_programs}")
                
                response = input("Do you want to add more sample data? (y/n): ")
                if response.lower() != 'y':
                    print("Cancelled by user.")
                    return
            
            # Sample Schools
            sample_schools = [
                {
                    'name': 'Sofia University "St. Kliment Ohridski"',
                    'location': 'Sofia, Bulgaria',
                    'description': 'The oldest and largest university in Bulgaria, offering diverse academic programs.',
                    'website': 'https://www.uni-sofia.bg',
                    'email': 'info@uni-sofia.bg',
                    'phone': '+359 2 9308 200'
                },
                {
                    'name': 'Technical University of Sofia',
                    'location': 'Sofia, Bulgaria', 
                    'description': 'Leading technical university specializing in engineering and technology.',
                    'website': 'https://www.tu-sofia.bg',
                    'email': 'info@tu-sofia.bg',
                    'phone': '+359 2 965 2111'
                },
                {
                    'name': 'University of National and World Economy',
                    'location': 'Sofia, Bulgaria',
                    'description': 'Premier business and economics university in Bulgaria.',
                    'website': 'https://www.unwe.bg',
                    'email': 'info@unwe.bg',
                    'phone': '+359 2 819 5050'
                },
                {
                    'name': 'Medical University of Sofia',
                    'location': 'Sofia, Bulgaria',
                    'description': 'Leading medical education institution in Bulgaria.',
                    'website': 'https://www.mu-sofia.bg',
                    'email': 'info@mu-sofia.bg',
                    'phone': '+359 2 952 0200'
                },
                {
                    'name': 'National Academy of Arts',
                    'location': 'Sofia, Bulgaria',
                    'description': 'Premier arts and design education institution.',
                    'website': 'https://nha.bg',
                    'email': 'info@nha.bg',
                    'phone': '+359 2 987 9797'
                },
                {
                    'name': 'Plovdiv University "Paisii Hilendarski"',
                    'location': 'Plovdiv, Bulgaria',
                    'description': 'Second largest university in Bulgaria with strong liberal arts programs.',
                    'website': 'https://www.uni-plovdiv.bg',
                    'email': 'info@uni-plovdiv.bg',
                    'phone': '+359 32 261 400'
                }
            ]
            
            print("🏫 Adding Schools...")
            schools = []
            for school_data in sample_schools:
                # Check if school already exists
                existing = School.query.filter_by(name=school_data['name']).first()
                if existing:
                    schools.append(existing)
                    print(f"   ✓ School already exists: {school_data['name']}")
                else:
                    school = School(**school_data)
                    db.session.add(school)
                    schools.append(school)
                    print(f"   + Added: {school_data['name']}")
            
            db.session.commit()
            print(f"✅ Schools completed: {len(schools)} total")
            
            # Sample Programs
            sample_programs = [
                # Sofia University
                {'name': 'Computer Science', 'school': 0, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 8000},
                {'name': 'Mathematics', 'school': 0, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 7000},
                {'name': 'Physics', 'school': 0, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 7500},
                {'name': 'Psychology', 'school': 0, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 7500},
                
                # Technical University
                {'name': 'Software Engineering', 'school': 1, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 9000},
                {'name': 'Mechanical Engineering', 'school': 1, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 8500},
                {'name': 'Electrical Engineering', 'school': 1, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 8500},
                {'name': 'Civil Engineering', 'school': 1, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 8000},
                
                # Business University
                {'name': 'Business Administration', 'school': 2, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 9500},
                {'name': 'Economics', 'school': 2, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 8500},
                {'name': 'International Business', 'school': 2, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 10000},
                {'name': 'Finance', 'school': 2, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 9000},
                
                # Medical University
                {'name': 'Medicine', 'school': 3, 'degree_type': 'Master', 'duration': '6 years', 'tuition_fee': 15000},
                {'name': 'Dentistry', 'school': 3, 'degree_type': 'Master', 'duration': '5 years', 'tuition_fee': 12000},
                {'name': 'Pharmacy', 'school': 3, 'degree_type': 'Master', 'duration': '5 years', 'tuition_fee': 10000},
                
                # Arts Academy
                {'name': 'Fine Arts', 'school': 4, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 6000},
                {'name': 'Graphic Design', 'school': 4, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 6500},
                {'name': 'Music Performance', 'school': 4, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 6000},
                
                # Plovdiv University
                {'name': 'Literature', 'school': 5, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 6500},
                {'name': 'History', 'school': 5, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 6500},
                {'name': 'Biology', 'school': 5, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 7000},
                {'name': 'Chemistry', 'school': 5, 'degree_type': 'Bachelor', 'duration': '4 years', 'tuition_fee': 7000},
            ]
            
            print("\n📚 Adding Programs...")
            programs = []
            for program_data in sample_programs:
                school_index = program_data.pop('school')
                school = schools[school_index]
                
                # Check if program already exists
                existing = Program.query.filter_by(
                    name=program_data['name'], 
                    school_id=school.id
                ).first()
                
                if existing:
                    programs.append(existing)
                    print(f"   ✓ Program already exists: {program_data['name']} at {school.name}")
                else:
                    program = Program(
                        school_id=school.id,
                        description=f"Comprehensive {program_data['name']} program at {school.name}",
                        **program_data
                    )
                    db.session.add(program)
                    programs.append(program)
                    print(f"   + Added: {program_data['name']} at {school.name}")
            
            db.session.commit()
            print(f"✅ Programs completed: {len(programs)} total")
            
            # Sample Survey (if none exists)
            print("\n📋 Checking for Survey...")
            existing_survey = Survey.query.filter_by(is_active=True).first()
            if not existing_survey:
                sample_questions = [
                    {
                        "id": "math_interest",
                        "question": "How interested are you in Mathematics?",
                        "type": "scale",
                        "scale": [1, 10]
                    },
                    {
                        "id": "science_interest", 
                        "question": "How interested are you in Science?",
                        "type": "scale",
                        "scale": [1, 10]
                    },
                    {
                        "id": "art_interest",
                        "question": "How interested are you in Arts?",
                        "type": "scale", 
                        "scale": [1, 10]
                    },
                    {
                        "id": "sports_interest",
                        "question": "How interested are you in Sports?",
                        "type": "scale",
                        "scale": [1, 10]
                    },
                    {
                        "id": "career_goal",
                        "question": "What is your career goal?",
                        "type": "text"
                    },
                    {
                        "id": "study_hours_per_day",
                        "question": "How many hours do you study per day?",
                        "type": "number"
                    },
                    {
                        "id": "grades_average",
                        "question": "What is your average grade?",
                        "type": "number"
                    }
                ]
                
                survey = Survey(
                    title="Academic Interest Survey",
                    description="Help us understand your academic interests and career goals",
                    questions=str(sample_questions).replace("'", '"'),
                    is_active=True
                )
                db.session.add(survey)
                db.session.commit()
                print("   + Added sample survey")
            else:
                print("   ✓ Survey already exists")
            
            print("\n" + "=" * 60)
            print("🎉 Sample Data Addition Complete!")
            
            # Final statistics
            final_schools = School.query.count()
            final_programs = Program.query.count()
            final_surveys = Survey.query.count()
            
            print("\n📊 Final Database Statistics:")
            print(f"   - Schools: {final_schools}")
            print(f"   - Programs: {final_programs}")
            print(f"   - Surveys: {final_surveys}")
            
            print("\n🚀 Ready to test!")
            print("Run: python test_recommendation_engine.py")
            
    except Exception as e:
        print(f"❌ Error adding sample data: {e}")
        return False
    
    return True

if __name__ == "__main__":
    add_sample_data() 