#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Create database tables directly without migrations
python -c "
from app import create_app
from app.models import db
app = create_app()
with app.app_context():
    db.create_all()
    print('Database tables created successfully')
" 