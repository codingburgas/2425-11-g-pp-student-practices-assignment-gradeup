# University Recommendation System

A Flask-based web application that helps students find their perfect university match using machine learning.

## Features

- User registration and authentication
- Survey system for data collection
- Machine learning-based university recommendations
- University database with detailed information
- Admin interface for content management

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd university-recommendation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
SECRET_KEY=your-secret-key
MAIL_SERVER=your-mail-server
MAIL_PORT=587
MAIL_USE_TLS=1
MAIL_USERNAME=your-email
MAIL_PASSWORD=your-password
```

5. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

6. Run the application:
```bash
flask run
```

The application will be available at `http://localhost:5000`

## Project Structure

```
university-recommendation/
├── app/
│   ├── __init__.py
│   ├── main/
│   ├── auth/
│   ├── admin/
│   ├── models/
│   └── templates/
├── migrations/
├── tests/
├── config.py
├── requirements.txt
└── README.md
```

## Development

- Follow PEP 8 style guide
- Write tests for new features
- Use meaningful commit messages
- Create feature branches for new development

## License

This project is licensed under the MIT License. 