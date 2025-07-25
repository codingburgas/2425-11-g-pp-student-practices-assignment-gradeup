import os
import sys
import json
from datetime import datetime
from sqlalchemy import create_engine, text, inspect
from app import create_app, db
from app.models import User, School, Program, Survey
from config import Config
from werkzeug.security import generate_password_hash

def create_tables():
    """Create all tables using SQLAlchemy models"""
    try:
        print("Creating tables using SQLAlchemy models...")
        app = create_app(Config)
        with app.app_context():
            db.create_all()
        print("All tables created successfully.")
        return True
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        return False

def create_admin_user():
    """Create an admin user"""
    try:
        app = create_app(Config)
        with app.app_context():
            
            admin = User.query.filter_by(username='admin').first()
            if admin is None:
                print("Creating admin user...")
                admin = User(
                    username='admin',
                    email='admin@example.com',
                    is_admin=True,
                    email_verified=True
                )
                admin.set_password('admin123')
                db.session.add(admin)
                db.session.commit()
                print("Admin user created successfully.")
            else:
                print("Admin user already exists.")
        return True
    except Exception as e:
        print(f"Error creating admin user: {str(e)}")
        return False

def create_sample_schools():
    """Create sample schools with programs"""
    try:
        app = create_app(Config)
        with app.app_context():
            
            if School.query.count() > 0:
                print("Schools already exist. Skipping...")
                return True
                
            print("Creating sample schools and programs...")
            
            
            schools_data = [
                {
                    "name": "Sofia University St. Kliment Ohridski",
                    "description": "The oldest higher education institution in Bulgaria with a strong focus on research and academic excellence.",
                    "location": "Sofia, Bulgaria",
                    "website": "https://www.uni-sofia.bg/",
                    "email": "info@uni-sofia.bg",
                    "phone": "+359 2 9308 200",
                    "admission_requirements": "High school diploma, entrance exam depending on faculty",
                    "programs_data": [
                        {
                            "name": "Computer Science",
                            "description": "Study of algorithms, programming languages, and computational systems",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Mathematics entrance exam",
                            "tuition_fee": 1800.00
                        },
                        {
                            "name": "Artificial Intelligence",
                            "description": "Study of machine learning, natural language processing, and computer vision",
                            "duration": "2 years",
                            "degree_type": "Master",
                            "admission_requirements": "Bachelor degree in Computer Science or related field",
                            "tuition_fee": 2500.00
                        },
                        {
                            "name": "Economics",
                            "description": "Study of economic systems, policies, and financial markets",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Entrance exam in Economics",
                            "tuition_fee": 1600.00
                        }
                    ]
                },
                {
                    "name": "Technical University of Sofia",
                    "description": "Leading technical university in Bulgaria specializing in engineering and technology",
                    "location": "Sofia, Bulgaria",
                    "website": "https://tu-sofia.bg/",
                    "email": "info@tu-sofia.bg",
                    "phone": "+359 2 965 2111",
                    "admission_requirements": "High school diploma, entrance exam in mathematics or physics",
                    "programs_data": [
                        {
                            "name": "Electrical Engineering",
                            "description": "Study of electrical systems, electronics, and power engineering",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Mathematics entrance exam",
                            "tuition_fee": 1900.00
                        },
                        {
                            "name": "Mechanical Engineering",
                            "description": "Study of mechanical systems, design, and manufacturing",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Mathematics or Physics entrance exam",
                            "tuition_fee": 1850.00
                        },
                        {
                            "name": "Computer Systems and Technologies",
                            "description": "Study of computer hardware, software systems, and networking",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Mathematics entrance exam",
                            "tuition_fee": 1950.00
                        }
                    ]
                },
                {
                    "name": "Medical University of Sofia",
                    "description": "Premier medical education and research institution in Bulgaria",
                    "location": "Sofia, Bulgaria",
                    "website": "https://mu-sofia.bg/",
                    "email": "info@mu-sofia.bg",
                    "phone": "+359 2 9172 501",
                    "admission_requirements": "High school diploma, entrance exams in Biology and Chemistry",
                    "programs_data": [
                        {
                            "name": "Medicine",
                            "description": "Study of human health and disease, leading to physician qualification",
                            "duration": "6 years",
                            "degree_type": "Master",
                            "admission_requirements": "Biology and Chemistry entrance exams",
                            "tuition_fee": 6500.00
                        },
                        {
                            "name": "Dental Medicine",
                            "description": "Study of oral health and dental procedures",
                            "duration": "5 years",
                            "degree_type": "Master",
                            "admission_requirements": "Biology and Chemistry entrance exams",
                            "tuition_fee": 5800.00
                        },
                        {
                            "name": "Pharmacy",
                            "description": "Study of pharmaceutical sciences and medication management",
                            "duration": "5 years",
                            "degree_type": "Master",
                            "admission_requirements": "Biology and Chemistry entrance exams",
                            "tuition_fee": 4500.00
                        }
                    ]
                },
                {
                    "name": "University of National and World Economy",
                    "description": "Bulgaria's oldest and largest economics university",
                    "location": "Sofia, Bulgaria",
                    "website": "https://www.unwe.bg/",
                    "email": "info@unwe.bg",
                    "phone": "+359 2 8195 211",
                    "admission_requirements": "High school diploma, entrance exam in selected subject",
                    "programs_data": [
                        {
                            "name": "Business Administration",
                            "description": "Study of organizational management and business operations",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Entrance exam in Economics or Mathematics",
                            "tuition_fee": 1700.00
                        },
                        {
                            "name": "Finance",
                            "description": "Study of financial systems, markets, and investment",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Entrance exam in Economics or Mathematics",
                            "tuition_fee": 1750.00
                        },
                        {
                            "name": "Marketing",
                            "description": "Study of market research, advertising, and consumer behavior",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Entrance exam in Economics",
                            "tuition_fee": 1650.00
                        }
                    ]
                },
                {
                    "name": "New Bulgarian University",
                    "description": "Private university known for its flexible and interdisciplinary programs",
                    "location": "Sofia, Bulgaria",
                    "website": "https://nbu.bg/",
                    "email": "info@nbu.bg",
                    "phone": "+359 2 8110 101",
                    "admission_requirements": "High school diploma, application process varies by program",
                    "programs_data": [
                        {
                            "name": "Computer Science",
                            "description": "Study of programming, software development, and computer systems",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Application and interview",
                            "tuition_fee": 3200.00
                        },
                        {
                            "name": "Psychology",
                            "description": "Study of human behavior, cognition, and mental processes",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Application and interview",
                            "tuition_fee": 2800.00
                        },
                        {
                            "name": "Mass Communications",
                            "description": "Study of media, journalism, and public relations",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Application and interview",
                            "tuition_fee": 2900.00
                        }
                    ]
                },
                {
                    "name": "Plovdiv University Paisii Hilendarski",
                    "description": "One of the largest universities in Bulgaria with a broad range of academic disciplines",
                    "location": "Plovdiv, Bulgaria",
                    "website": "https://uni-plovdiv.bg/",
                    "email": "info@uni-plovdiv.bg",
                    "phone": "+359 32 261 261",
                    "admission_requirements": "High school diploma, entrance exams vary by faculty",
                    "programs_data": [
                        {
                            "name": "Computer Science",
                            "description": "Study of algorithms, programming languages, and software development",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Mathematics entrance exam",
                            "tuition_fee": 1750.00
                        },
                        {
                            "name": "Biology",
                            "description": "Study of living organisms, ecology, and biological systems",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Biology entrance exam",
                            "tuition_fee": 1600.00
                        },
                        {
                            "name": "Linguistics",
                            "description": "Study of language structure, acquisition, and communication",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Bulgarian language entrance exam",
                            "tuition_fee": 1400.00
                        }
                    ]
                },
                {
                    "name": "American University in Bulgaria",
                    "description": "Liberal arts university offering American-style education in English",
                    "location": "Blagoevgrad, Bulgaria",
                    "website": "https://www.aubg.edu/",
                    "email": "info@aubg.edu",
                    "phone": "+359 73 888 211",
                    "admission_requirements": "High school diploma, SAT/ACT scores, English proficiency test",
                    "programs_data": [
                        {
                            "name": "Computer Science",
                            "description": "Study of programming, algorithms, and computational systems",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "SAT/ACT scores, English proficiency",
                            "tuition_fee": 8900.00
                        },
                        {
                            "name": "Business Administration",
                            "description": "Study of management, marketing, and business operations",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "SAT/ACT scores, English proficiency",
                            "tuition_fee": 8500.00
                        },
                        {
                            "name": "Political Science",
                            "description": "Study of government systems, international relations, and public policy",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "SAT/ACT scores, English proficiency",
                            "tuition_fee": 8200.00
                        }
                    ]
                },
                {
                    "name": "Varna University of Management",
                    "description": "International university specializing in business, hospitality and tourism",
                    "location": "Varna, Bulgaria",
                    "website": "https://vum.bg/",
                    "email": "info@vum.bg",
                    "phone": "+359 52 300 680",
                    "admission_requirements": "High school diploma, English proficiency test",
                    "programs_data": [
                        {
                            "name": "International Business Management",
                            "description": "Study of global business operations and management strategies",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "English proficiency, application",
                            "tuition_fee": 4200.00
                        },
                        {
                            "name": "Hospitality Management",
                            "description": "Study of hotel operations, tourism, and guest services",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "English proficiency, application",
                            "tuition_fee": 3800.00
                        },
                        {
                            "name": "Software Systems and Technologies",
                            "description": "Study of software development, programming, and IT systems",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "English proficiency, application",
                            "tuition_fee": 4500.00
                        }
                    ]
                },
                {
                    "name": "Burgas Free University",
                    "description": "Private university with a focus on business, law, and computer science",
                    "location": "Burgas, Bulgaria",
                    "website": "https://www.bfu.bg/",
                    "email": "info@bfu.bg",
                    "phone": "+359 56 900 400",
                    "admission_requirements": "High school diploma, entrance exam for some programs",
                    "programs_data": [
                        {
                            "name": "Informatics and Computer Science",
                            "description": "Study of programming, algorithms, and IT systems",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Entrance exam in Mathematics or IT",
                            "tuition_fee": 2800.00
                        },
                        {
                            "name": "Law",
                            "description": "Study of legal systems, jurisprudence, and legal practice",
                            "duration": "5 years",
                            "degree_type": "Master",
                            "admission_requirements": "Entrance exam in Bulgarian language and literature",
                            "tuition_fee": 3500.00
                        },
                        {
                            "name": "Business Administration",
                            "description": "Study of management, marketing, and business operations",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Entrance exam in Economics",
                            "tuition_fee": 2600.00
                        }
                    ]
                },
                {
                    "name": "Ruse University Angel Kanchev",
                    "description": "Technical university specializing in engineering, computer science, and economics",
                    "location": "Ruse, Bulgaria",
                    "website": "https://www.uni-ruse.bg/",
                    "email": "info@uni-ruse.bg",
                    "phone": "+359 82 888 211",
                    "admission_requirements": "High school diploma, entrance exam in Mathematics or subject test",
                    "programs_data": [
                        {
                            "name": "Computer Systems and Technologies",
                            "description": "Study of computer hardware, networking, and system design",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Mathematics entrance exam",
                            "tuition_fee": 1800.00
                        },
                        {
                            "name": "Mechanical Engineering",
                            "description": "Study of mechanical systems, manufacturing, and design",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Mathematics or Physics entrance exam",
                            "tuition_fee": 1750.00
                        },
                        {
                            "name": "Economics",
                            "description": "Study of economic systems, market analysis, and financial management",
                            "duration": "4 years",
                            "degree_type": "Bachelor",
                            "admission_requirements": "Mathematics or Economics entrance exam",
                            "tuition_fee": 1550.00
                        }
                    ]
                }
            ]
            
            
            total_programs = 0
            for school_data in schools_data:
                
                school_dict = dict(school_data)
                programs_data = school_dict.pop('programs_data')
                
                
                school = School(**school_dict)
                db.session.add(school)
                db.session.flush()  
                
                
                for program_data in programs_data:
                    program = Program(school_id=school.id, **program_data)
                    db.session.add(program)
                    total_programs += 1
            
            db.session.commit()
            print(f"Created {len(schools_data)} schools with {total_programs} programs.")
        return True
    except Exception as e:
        print(f"Error creating sample schools: {str(e)}")
        return False

def create_sample_survey():
    """Create a sample survey for data collection"""
    try:
        app = create_app(Config)
        with app.app_context():
            
            if Survey.query.count() > 0:
                print("Survey already exists. Skipping...")
                return True
                
            print("Creating sample survey...")
            
            survey_questions = [
                {
                    "id": 1,
                    "text": "What subjects do you enjoy the most?",
                    "type": "multiple_choice",
                    "options": ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science", "History", "Literature", "Languages", "Arts", "Economics", "Psychology", "Philosophy"]
                },
                {
                    "id": 2,
                    "text": "What type of career are you interested in?",
                    "type": "multiple_choice",
                    "options": ["Technology", "Science", "Medicine", "Business", "Law", "Education", "Arts", "Engineering", "Social Services", "Government"]
                },
                {
                    "id": 3,
                    "text": "How important is university location to you?",
                    "type": "rating",
                    "min": 1,
                    "max": 5
                },
                {
                    "id": 4,
                    "text": "Do you prefer theoretical or practical learning?",
                    "type": "slider",
                    "min": 1,
                    "max": 5,
                    "labels": ["Theoretical", "Balanced", "Practical"]
                },
                {
                    "id": 5,
                    "text": "What program length do you prefer?",
                    "type": "multiple_choice",
                    "options": ["3 years", "4 years", "5 years", "6 years"]
                },
                {
                    "id": 6,
                    "text": "How important is employment rate after graduation?",
                    "type": "rating",
                    "min": 1,
                    "max": 5
                },
                {
                    "id": 7,
                    "text": "What teaching language do you prefer?",
                    "type": "multiple_choice",
                    "options": ["Bulgarian", "English", "Other", "Doesn't matter"]
                },
                {
                    "id": 8,
                    "text": "What are your average grades in high school?",
                    "type": "multiple_choice",
                    "options": ["Below 4.0", "4.0-4.5", "4.5-5.0", "5.0-5.5", "5.5-6.0"]
                },
                {
                    "id": 9,
                    "text": "What extracurricular activities do you enjoy?",
                    "type": "multiple_select",
                    "options": ["Sports", "Music", "Art", "Programming", "Volunteering", "Student Government", "Debate", "None"]
                },
                {
                    "id": 10,
                    "text": "What skills would you like to develop?",
                    "type": "multiple_select",
                    "options": ["Problem Solving", "Critical Thinking", "Communication", "Leadership", "Teamwork", "Technical Skills", "Creative Thinking", "Research Skills"]
                }
            ]
            
            survey = Survey(
                title="Educational Preferences Survey",
                description="This survey helps us understand your educational preferences and career goals to recommend suitable academic programs.",
                questions=json.dumps(survey_questions),
                is_active=True
            )
            
            db.session.add(survey)
            db.session.commit()
            print("Sample survey created successfully.")
        return True
    except Exception as e:
        print(f"Error creating sample survey: {str(e)}")
        return False

def verify_admin_email():
    app = create_app(Config)
    with app.app_context():
        admin = User.query.filter_by(username='admin').first()
        if admin and not admin.email_verified:
            admin.email_verified = True
            db.session.commit()
            print("Admin user email marked as verified (via setup.py).")

def main():
    """Main execution function"""
    print("=" * 50)
    print("School Recommendation System - Database Setup")
    print("=" * 50)
    
    # Create tables
    if not create_tables():
        print("Table creation failed. Exiting...")
        sys.exit(1)
    
    # Create admin user
    if not create_admin_user():
        print("Admin user creation failed. Exiting...")
        sys.exit(1)
    
    # Create sample schools
    if not create_sample_schools():
        print("Sample schools creation failed. Exiting...")
        sys.exit(1)
    
    # Create sample survey
    if not create_sample_survey():
        print("Sample survey creation failed. Exiting...")
        sys.exit(1)
    
    print("=" * 50)
    print("Database setup completed successfully!")
    print("Database is clean and ready for real user survey responses!")
    print("You can now start the application.")
    print("=" * 50)
    print("Admin credentials:")
    print("Username: admin")
    print("Password: admin123")
    print("=" * 50)

if __name__ == "__main__":
    main()
    verify_admin_email() 