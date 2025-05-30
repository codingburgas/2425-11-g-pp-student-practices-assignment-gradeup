# 🎓 School Recommendation System

A web application that helps students find the perfect educational institution based on their preferences and needs.

---

## 🚀 Features

- 🔐User authentication (registration, login, profile management)  
- 🎓Student dashboard with personalized recommendations  
- 🛠️ Admin panel for user and system management  
- 🏫 School and program browsing  
- ⭐ Favorites system to save and compare schools

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/school-recommendation-system.git
   cd school-recommendation-system
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**  
     ```bash
     venv\Scripts\activate
     ```
   - **MacOS/Linux:**  
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**  
   Create a `.env` file and configure your environment settings.

6. **Initialize the database:**
   ```bash
   flask db upgrade
   ```

7. **Run the application:**
   ```bash
   flask run
   ```

---

## 📁 Project Structure

```
school-recommendation-system/
│
├── app/                  # Application package
│   ├── main/             # General pages blueprint
│   ├── auth/             # Authentication blueprint
│   ├── admin/            # Admin panel blueprint
│   ├── static/           # Static files (CSS, JS, images)
│   └── templates/        # Jinja2 templates
│
├── migrations/           # Database migration scripts
├── models.py             # Database models
├── config.py             # Configuration settings
├── app.py                # Entry point of the application
└── requirements.txt      # Python dependencies
```

---

## 💻 Technology Stack

- **Flask** – Web framework  
- **SQLAlchemy** – ORM for database operations  
- **Flask-Login** – User session management  
- **Bootstrap** – Frontend UI framework  
- **SQL Server** – Configurable backend database

