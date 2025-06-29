#!/usr/bin/env python3
"""
Database Connection Diagnostic Script
Run this to check if your database connection is working properly.
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_database_connection():
    """Test the database connection and provide diagnostic information."""
    
    print("🔍 Database Connection Diagnostic")
    print("=" * 50)
    
    # Check environment variables
    database_url = os.environ.get('DATABASE_URL')
    print(f"DATABASE_URL environment variable: {'✓ Set' if database_url else '✗ Not set'}")
    
    if database_url:
        # Hide password in the output
        safe_url = database_url
        if '@' in database_url:
            parts = database_url.split('@')
            if ':' in parts[0]:
                user_pass = parts[0].split(':')
                if len(user_pass) > 2:  # has password
                    safe_url = f"{user_pass[0]}:{user_pass[1]}:***@{parts[1]}"
        print(f"Connection string: {safe_url}")
    
    # Try to create engine and connect
    try:
        if not database_url:
            print("❌ No DATABASE_URL found. Please set it in your Azure App Service configuration.")
            return False
            
        # Fix postgres:// to postgresql:// if needed
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
            print("📝 Fixed postgres:// to postgresql://")
        
        print("\n🔄 Creating database engine...")
        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=True  # This will show SQL queries
        )
        
        print("🔄 Testing connection...")
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1 as test"))
            test_value = result.fetchone()[0]
            if test_value == 1:
                print("✅ Database connection successful!")
                
                # Test if we can access our tables
                print("\n🔄 Checking database schema...")
                try:
                    tables_result = connection.execute(text("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        ORDER BY table_name
                    """))
                    tables = [row[0] for row in tables_result.fetchall()]
                    
                    if tables:
                        print(f"✅ Found {len(tables)} tables: {', '.join(tables)}")
                    else:
                        print("⚠️  No tables found. You may need to run migrations.")
                        print("   Run: flask db upgrade")
                        
                except Exception as e:
                    print(f"⚠️  Could not check tables: {e}")
                
                return True
            else:
                print("❌ Database connection test failed")
                return False
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n🔧 Common solutions:")
        print("1. Check if your Azure PostgreSQL server is running")
        print("2. Verify the connection string format")
        print("3. Check firewall rules (allow Azure services)")  
        print("4. Verify username/password are correct")
        print("5. Make sure the database name exists")
        return False

def check_flask_app():
    """Check if Flask app can initialize properly."""
    print("\n🔍 Flask App Initialization Check")
    print("=" * 50)
    
    try:
        from app import create_app, db
        print("✅ Successfully imported Flask app")
        
        app = create_app()
        print("✅ Successfully created Flask app")
        
        with app.app_context():
            print("🔄 Testing database initialization...")
            # Try to get database info
            db.create_all()  # This creates tables if they don't exist
            print("✅ Database initialization successful")
            
            # Try to query a simple table count
            from sqlalchemy import text
            result = db.session.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"))
            table_count = result.scalar()
            print(f"✅ Found {table_count} tables in database")
            
        return True
        
    except Exception as e:
        print(f"❌ Flask app initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Database Diagnostics...\n")
    
    # Test basic connection
    db_ok = test_database_connection()
    
    if db_ok:
        # Test Flask app
        flask_ok = check_flask_app()
        
        if flask_ok:
            print("\n🎉 All checks passed! Your database should be working.")
        else:
            print("\n⚠️  Database connects but Flask app has issues.")
    else:
        print("\n❌ Database connection failed. Fix connection issues first.")
    
    print("\n📝 Next steps if there are issues:")
    print("1. Set DATABASE_URL in Azure App Service Configuration")
    print("2. Run 'flask db upgrade' to create/update database schema")
    print("3. Check Azure PostgreSQL firewall settings")
    print("4. Verify your connection string format") 