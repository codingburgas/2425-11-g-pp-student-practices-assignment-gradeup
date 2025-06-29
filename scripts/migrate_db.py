#!/usr/bin/env python3
"""
Database Migration Script for Azure
This script will create all necessary database tables.
"""

import os
from flask_migrate import upgrade
from app import create_app, db

def deploy():
    """Run deployment tasks."""
    app = create_app()
    
    with app.app_context():
        # Create database tables
        print("Creating database tables...")
        db.create_all()
        
        # Run any pending migrations
        print("Running database migrations...")
        try:
            upgrade()
            print("✅ Database migrations completed successfully!")
        except Exception as e:
            print(f"Migration warning: {e}")
            print("This might be normal if no migrations exist yet.")
        
        # Verify tables were created
        from sqlalchemy import text
        result = db.session.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"))
        table_count = result.scalar()
        print(f"✅ Database now has {table_count} tables")

if __name__ == '__main__':
    deploy() 