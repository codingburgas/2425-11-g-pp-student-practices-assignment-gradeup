import pyodbc
import os
import json
from datetime import datetime
from werkzeug.security import generate_password_hash
import sys

def create_database():
    """Create the SchoolRecommendation database if it doesn't exist."""
    try:
        # Connect to the master database to create our database
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=localhost;'
            'DATABASE=master;'
            'Trusted_Connection=yes;'
        )
        
        cursor = conn.cursor()
        
        # Check if database already exists
        cursor.execute("SELECT name FROM sys.databases WHERE name = 'SchoolRecommendation'")
        result = cursor.fetchone()
        
        if not result:
            print("Creating SchoolRecommendation database...")
            cursor.execute("CREATE DATABASE SchoolRecommendation")
            print("Database created successfully!")
        else:
            print("Database 'SchoolRecommendation' already exists.")
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

def create_tables():
    """Create all tables for the application."""
    try:
        # Connect to our database
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=localhost;'
            'DATABASE=SchoolRecommendation;'
            'Trusted_Connection=yes;'
        )
        
        cursor = conn.cursor()
        
        # Create users table
        print("Creating users table...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[users]') AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[users] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [username] NVARCHAR(64) NOT NULL UNIQUE,
                    [email] NVARCHAR(120) NOT NULL UNIQUE,
                    [password_hash] NVARCHAR(128),
                    [is_admin] BIT DEFAULT 0,
                    [created_at] DATETIME DEFAULT GETDATE(),
                    [profile_picture] NVARCHAR(255) NULL
                )
                
                CREATE INDEX ix_users_username ON [dbo].[users] ([username])
                CREATE INDEX ix_users_email ON [dbo].[users] ([email])
            END
        """)
        
        # Create schools table
        print("Creating schools table...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[schools]') AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[schools] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [name] NVARCHAR(100) NOT NULL,
                    [description] NVARCHAR(MAX) NULL,
                    [location] NVARCHAR(100) NOT NULL,
                    [website] NVARCHAR(255) NULL,
                    [email] NVARCHAR(100) NULL,
                    [phone] NVARCHAR(20) NULL,
                    [logo] NVARCHAR(255) NULL,
                    [admission_requirements] NVARCHAR(MAX) NULL,
                    [created_at] DATETIME DEFAULT GETDATE(),
                    [updated_at] DATETIME DEFAULT GETDATE()
                )
                
                CREATE INDEX ix_schools_name ON [dbo].[schools] ([name])
            END
        """)
        
        # Create programs table
        print("Creating programs table...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[programs]') AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[programs] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [name] NVARCHAR(100) NOT NULL,
                    [description] NVARCHAR(MAX) NULL,
                    [duration] NVARCHAR(50) NULL,
                    [degree_type] NVARCHAR(50) NOT NULL,
                    [admission_requirements] NVARCHAR(MAX) NULL,
                    [tuition_fee] FLOAT NULL,
                    [created_at] DATETIME DEFAULT GETDATE(),
                    [updated_at] DATETIME DEFAULT GETDATE(),
                    [school_id] INT NOT NULL,
                    CONSTRAINT [FK_programs_schools] FOREIGN KEY ([school_id]) REFERENCES [dbo].[schools] ([id])
                )
                
                CREATE INDEX ix_programs_name ON [dbo].[programs] ([name])
            END
        """)
        
        # Create surveys table
        print("Creating surveys table...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[surveys]') AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[surveys] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [title] NVARCHAR(100) NOT NULL,
                    [description] NVARCHAR(MAX) NULL,
                    [questions] NVARCHAR(MAX) NOT NULL,
                    [is_active] BIT DEFAULT 1,
                    [created_at] DATETIME DEFAULT GETDATE(),
                    [updated_at] DATETIME DEFAULT GETDATE()
                )
            END
        """)
        
        # Create survey_responses table
        print("Creating survey_responses table...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[survey_responses]') AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[survey_responses] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [answers] NVARCHAR(MAX) NOT NULL,
                    [created_at] DATETIME DEFAULT GETDATE(),
                    [user_id] INT NOT NULL,
                    [survey_id] INT NOT NULL,
                    CONSTRAINT [FK_survey_responses_users] FOREIGN KEY ([user_id]) REFERENCES [dbo].[users] ([id]),
                    CONSTRAINT [FK_survey_responses_surveys] FOREIGN KEY ([survey_id]) REFERENCES [dbo].[surveys] ([id])
                )
            END
        """)
        
        # Create recommendations table
        print("Creating recommendations table...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[recommendations]') AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[recommendations] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [score] FLOAT NOT NULL,
                    [created_at] DATETIME DEFAULT GETDATE(),
                    [survey_response_id] INT NOT NULL,
                    [program_id] INT NOT NULL,
                    CONSTRAINT [FK_recommendations_survey_responses] FOREIGN KEY ([survey_response_id]) REFERENCES [dbo].[survey_responses] ([id]),
                    CONSTRAINT [FK_recommendations_programs] FOREIGN KEY ([program_id]) REFERENCES [dbo].[programs] ([id])
                )
            END
        """)
        
        # Create favorites table
        print("Creating favorites table...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[favorites]') AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[favorites] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [created_at] DATETIME DEFAULT GETDATE(),
                    [user_id] INT NOT NULL,
                    [school_id] INT NOT NULL,
                    CONSTRAINT [FK_favorites_users] FOREIGN KEY ([user_id]) REFERENCES [dbo].[users] ([id]),
                    CONSTRAINT [FK_favorites_schools] FOREIGN KEY ([school_id]) REFERENCES [dbo].[schools] ([id]),
                    CONSTRAINT [UQ_favorite_user_school] UNIQUE ([user_id], [school_id])
                )
            END
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("All tables created successfully!")
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

def add_sample_data():
    """Add sample data to the database for testing."""
    try:
        # Connect to our database
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=localhost;'
            'DATABASE=SchoolRecommendation;'
            'Trusted_Connection=yes;'
        )
        
        cursor = conn.cursor()
        
        # Check if we already have data
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count > 0:
            print("Sample data already exists. Skipping...")
            return True
        
        print("Adding sample data...")
        
        # Add admin user
        admin_password_hash = generate_password_hash("admin123")
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, is_admin)
            VALUES (?, ?, ?, ?)
        """, ("admin", "admin@example.com", admin_password_hash, True))
        
        # Add regular user
        user_password_hash = generate_password_hash("user123")
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, is_admin)
            VALUES (?, ?, ?, ?)
        """, ("user", "user@example.com", user_password_hash, False))
        
        # Add sample schools
        schools = [
            ("Sofia University", "The oldest higher education institution in Bulgaria", "Sofia", 
             "https://www.uni-sofia.bg", "info@uni-sofia.bg", "+359 2 9308 200"),
            ("Technical University of Sofia", "The largest technical university in Bulgaria", "Sofia", 
             "https://www.tu-sofia.bg", "tu@tu-sofia.bg", "+359 2 965 3241"),
            ("Medical University of Sofia", "Leading medical university in Bulgaria", "Sofia", 
             "https://mu-sofia.bg", "mail@mu-sofia.bg", "+359 2 9172 501")
        ]
        
        for school in schools:
            cursor.execute("""
                INSERT INTO schools (name, description, location, website, email, phone)
                VALUES (?, ?, ?, ?, ?, ?)
            """, school)
            
        # Get school IDs
        cursor.execute("SELECT id FROM schools")
        school_ids = [row[0] for row in cursor.fetchall()]
        
        # Add sample programs
        programs = [
            ("Computer Science", "Bachelor's degree in Computer Science", "4 years", "Bachelor", 
             "High school diploma with good grades in Mathematics", 3500, school_ids[1]),
            ("Medicine", "Medical Doctor degree", "6 years", "Master", 
             "High school diploma with excellent grades in Biology and Chemistry", 5000, school_ids[2]),
            ("Business Administration", "Bachelor's degree in Business Administration", "4 years", 
             "Bachelor", "High school diploma", 3000, school_ids[0])
        ]
        
        for program in programs:
            cursor.execute("""
                INSERT INTO programs (name, description, duration, degree_type, admission_requirements, tuition_fee, school_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, program)
        
        # Add sample survey
        questions = [
            {"id": 1, "text": "What subjects do you enjoy most?", "type": "multiple_choice", 
             "options": ["Math", "Biology", "Chemistry", "Physics", "Literature", "History", "Arts"]},
            {"id": 2, "text": "How important is location to you?", "type": "rating", "min": 1, "max": 5},
            {"id": 3, "text": "What type of career are you interested in?", "type": "text"},
            {"id": 4, "text": "Do you prefer theoretical or practical learning?", "type": "choice", 
             "options": ["Theoretical", "Practical", "Balanced"]},
            {"id": 5, "text": "What is your budget for tuition fees per year?", "type": "number"}
        ]
        
        cursor.execute("""
            INSERT INTO surveys (title, description, questions)
            VALUES (?, ?, ?)
        """, ("Education Preference Survey", "Survey to determine your educational preferences", json.dumps(questions)))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("Sample data added successfully!")
        return True
    except Exception as e:
        print(f"Error adding sample data: {e}")
        return False

def main():
    print("Setting up SchoolRecommendation database...")
    
    # Check for ODBC driver
    try:
        pyodbc.drivers()
    except:
        print("Error: pyodbc module not installed. Please install it using 'pip install pyodbc'")
        return
    
    odbc_drivers = pyodbc.drivers()
    if 'ODBC Driver 17 for SQL Server' not in odbc_drivers:
        print("Warning: ODBC Driver 17 for SQL Server not found. Available drivers:")
        for driver in odbc_drivers:
            print(f"  - {driver}")
        print("\nPlease install the Microsoft ODBC Driver for SQL Server from:")
        print("https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server")
        
        use_different_driver = input("Would you like to use a different driver? (y/n): ")
        if use_different_driver.lower() == 'y':
            print("\nAvailable drivers:")
            for i, driver in enumerate(odbc_drivers):
                print(f"{i+1}. {driver}")
            
            driver_index = int(input("Enter the number of the driver to use: ")) - 1
            if 0 <= driver_index < len(odbc_drivers):
                # Update the driver in config.py
                with open('config.py', 'r') as f:
                    config_content = f.read()
                
                new_driver = odbc_drivers[driver_index]
                config_content = config_content.replace("ODBC Driver 17 for SQL Server", new_driver)
                
                with open('config.py', 'w') as f:
                    f.write(config_content)
                
                print(f"Updated config.py to use {new_driver}")
            else:
                print("Invalid driver selection. Exiting.")
                return
        else:
            print("Continuing with ODBC Driver 17 for SQL Server, but setup might fail.")
    
    # Create the database
    if not create_database():
        return
    
    # Create tables
    if not create_tables():
        return
    
    # Add sample data
    add_sample_data()
    
    print("\nSetup completed successfully!")
    print("\nYou can now run the application using:")
    print("flask run")

if __name__ == "__main__":
    main() 