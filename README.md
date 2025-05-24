# School Recommendation System

A web application that helps students find the perfect educational institution based on their preferences and needs.

## Features

- User authentication (registration, login, profile management)
- Student dashboard with personalized recommendations
- Admin panel for user management
- School and program browsing
- Favorites system to save and compare schools

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - MacOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Set up environment variables in a `.env` file
6. Initialize the database: `flask db upgrade`
7. Run the application: `flask run`

## Project Structure

- `app/` - Application package
  - `main/` - Main blueprint for general pages
  - `auth/` - Authentication blueprint for user management
  - `admin/` - Admin blueprint for administration functions
  - `static/` - Static files (CSS, JS, images)
  - `templates/` - Jinja2 templates
  - `models.py` - Database models
- `migrations/` - Database migration scripts
- `config.py` - Configuration settings
- `app.py` - Application entry point

## Technology Stack

- Flask - Web framework
- SQLAlchemy - ORM for database operations
- Flask-Login - User session management
- Bootstrap - Frontend framework
- SQLServer - Database (configurable)

---
## 🚀 Features
