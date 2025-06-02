# GradeUP Azure Deployment Guide

## Overview
This guide will help you deploy the GradeUP Flask application to Azure App Service with Azure SQL Database for student survey data collection.

## Prerequisites
- Azure subscription
- Azure CLI installed (optional but recommended)
- Git repository with your code

## Step 1: Create Azure SQL Database

### 1.1 Create SQL Server
1. Go to Azure Portal → Create a resource → SQL Server
2. Fill in the details:
   - Server name: `gradeup-server` (or your preferred name)
   - Admin login: `gradeup_admin`
   - Password: Create a strong password
   - Location: Choose closest to your students
3. Click "Create"

### 1.2 Create SQL Database
1. Go to your SQL Server → Add database
2. Database name: `gradeup_db`
3. Choose Basic pricing tier (sufficient for student surveys)
4. Click "Create"

### 1.3 Configure Firewall
1. Go to SQL Server → Networking
2. Add rule to allow Azure services: ✅ Allow Azure services and resources to access this server
3. Add your IP for management access
4. Save

## Step 2: Create Azure App Service

### 2.1 Create Web App
1. Go to Azure Portal → Create a resource → Web App
2. Fill in details:
   - App name: `gradeup-surveys` (must be globally unique)
   - Runtime stack: Python 3.12
   - Operating System: Linux
   - Region: Same as your database
   - Pricing: B1 Basic (recommended for student load)
3. Click "Create"

## Step 3: Configure Application Settings

### 3.1 Add Environment Variables
Go to App Service → Configuration → Application Settings and add:

```
FLASK_APP = startup.py
FLASK_ENV = production
SECRET_KEY = [Generate a 32-character random string]
DATABASE_URL = mssql+pyodbc://gradeup_admin:[YOUR_PASSWORD]@gradeup-server.database.windows.net/gradeup_db?driver=ODBC+Driver+17+for+SQL+Server
WEBSITE_HOSTNAME = gradeup-surveys.azurewebsites.net
```

### 3.2 Optional Email Settings (for notifications)
```
MAIL_SERVER = smtp.gmail.com
MAIL_PORT = 587
MAIL_USE_TLS = 1
MAIL_USERNAME = your-email@gmail.com
MAIL_PASSWORD = your-app-password
```

## Step 4: Deploy Application

### 4.1 Method 1: From GitHub (Recommended)
1. Go to App Service → Deployment Center
2. Select "GitHub" as source
3. Authorize and select your repository
4. Select branch: `feature/design-implementation`
5. Click "Save" - Azure will automatically deploy

### 4.2 Method 2: ZIP Deploy
1. Create a ZIP file of your project
2. Go to App Service → Development Tools → Advanced Tools → Go
3. Navigate to /home/site/wwwroot
4. Drag and drop your ZIP file

### 4.3 Method 3: Azure CLI
```bash
# Login to Azure
az login

# Set subscription (if you have multiple)
az account set --subscription "Your Subscription Name"

# Deploy from local Git
az webapp deployment source config-local-git --name gradeup-surveys --resource-group your-resource-group

# Add Azure as Git remote and push
git remote add azure [git-url-from-previous-command]
git push azure feature/design-implementation:master
```

## Step 5: Initialize Database

### 5.1 Automatic Initialization
The `startup.py` file will automatically create database tables on first run.

### 5.2 Manual Initialization (if needed)
1. Go to App Service → SSH → Go
2. Run:
```bash
cd /home/site/wwwroot
python startup.py
```

## Step 6: Test Your Deployment

1. Visit your app URL: `https://gradeup-surveys.azurewebsites.net`
2. Test student registration and login
3. Test survey functionality
4. Check database connectivity

## Step 7: Create Admin User

### 7.1 Through App Service Console
1. Go to App Service → SSH
2. Run:
```python
from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash

app = create_app()
with app.app_context():
    admin = User(
        username='admin',
        email='admin@yourdomain.com',
        password_hash=generate_password_hash('your-admin-password'),
        is_admin=True
    )
    db.session.add(admin)
    db.session.commit()
    print("Admin user created!")
```

## Step 8: Student Access Instructions

### For Students:
1. Visit: `https://gradeup-surveys.azurewebsites.net`
2. Click "Register" to create account
3. Complete profile information
4. Take the survey from Dashboard
5. View recommendations

### Available Features for Students:
- ✅ Home page
- ✅ User registration/login
- ✅ Dashboard
- ✅ Survey taking
- ✅ Profile management
- ❌ Admin features (hidden)
- ❌ Universities/Specialties pages (hidden)

## Step 9: Monitor and Maintain

### 9.1 Monitor Application
- Go to App Service → Monitoring → Metrics
- Check response times, errors, and resource usage

### 9.2 View Logs
- Go to App Service → Monitoring → Log stream
- Check for any errors or issues

### 9.3 Scale if Needed
- Go to App Service → Scale up (for better performance)
- Or Scale out (for more instances)

## Troubleshooting

### Common Issues:

1. **Database Connection Errors**
   - Check firewall settings
   - Verify connection string format
   - Ensure Azure services access is enabled

2. **Module Import Errors**
   - Verify requirements.txt is complete
   - Check Python version compatibility

3. **Static Files Not Loading**
   - Ensure static files are in the repository
   - Check Flask static file configuration

4. **Survey Not Working**
   - Check browser console for JavaScript errors
   - Verify CSRF token configuration

### Getting Support:
- Check Application Insights for detailed error logs
- Review Azure App Service logs
- Contact Azure support if needed

## Security Considerations

1. **Database Security**
   - Use strong passwords
   - Enable SSL connections
   - Regular security updates

2. **Application Security**
   - Keep Flask and dependencies updated
   - Use HTTPS only
   - Implement proper session management

3. **Data Privacy**
   - Follow GDPR/local privacy laws
   - Implement data retention policies
   - Secure student data

## Cost Optimization

- **Basic Tier** should be sufficient for student surveys
- **Monitor usage** and scale down if needed
- **Set up billing alerts** to avoid unexpected costs
- **Consider reserved instances** for long-term use

---

**Note:** This deployment focuses on the essential student survey functionality while hiding advanced admin features. Students will only see: Home, Survey, Login, Register, Dashboard, and Profile pages. 