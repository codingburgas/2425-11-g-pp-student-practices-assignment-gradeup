from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_mail import Mail
from flask_wtf.csrf import CSRFProtect
from config import Config
from datetime import datetime, timezone
import pytz

db = SQLAlchemy()
migrate = Migrate()
login = LoginManager()
login.login_view = 'auth.login'
login.login_message = 'Please log in to access this page.'
mail = Mail()
bootstrap = Bootstrap()
csrf = CSRFProtect()

def create_app(config_class=Config):
    app = Flask(__name__, 
                template_folder='../templates',  
                static_folder='static')
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    login.init_app(app)
    login.login_view = 'auth.login'
    login.login_message = 'Please log in to access this page.'
    mail.init_app(app)
    bootstrap.init_app(app)
    csrf.init_app(app)

    
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    from app.admin import bp as admin_bp
    app.register_blueprint(admin_bp, url_prefix='/admin')
    
    from app.data_collection import bp as data_collection_bp
    app.register_blueprint(data_collection_bp, url_prefix='/data')

    
    from app.ml.blueprint import ml_bp
    app.register_blueprint(ml_bp)
    
    # Register Advanced Prediction System blueprint
    from app.ml.prediction_blueprint import prediction_bp
    app.register_blueprint(prediction_bp)
    
    # Register Recommendation Engine blueprint
    from app.ml.recommendation_blueprint import recommendation_bp
    app.register_blueprint(recommendation_bp)

    # Add timezone conversion filters
    @app.template_filter('localtime')
    def localtime_filter(utc_datetime, timezone_name='Europe/Sofia'):
        """Convert UTC datetime to local timezone"""
        if utc_datetime is None:
            return 'N/A'
        
        # Assume the datetime stored is in UTC
        if utc_datetime.tzinfo is None:
            utc_datetime = utc_datetime.replace(tzinfo=pytz.UTC)
        
        # Convert to specified timezone (default to Sofia, Bulgaria)
        local_tz = pytz.timezone(timezone_name)
        local_datetime = utc_datetime.astimezone(local_tz)
        
        return local_datetime
    
    @app.template_filter('formatlocal')
    def formatlocal_filter(utc_datetime, fmt='%B %d, %Y at %I:%M %p', timezone_name='Europe/Sofia'):
        """Convert UTC datetime to local timezone and format it"""
        if utc_datetime is None:
            return 'N/A'
        
        local_datetime = localtime_filter(utc_datetime, timezone_name)
        if local_datetime == 'N/A':
            return 'N/A'
            
        return local_datetime.strftime(fmt)

    # Register error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('500.html'), 500

    # Initialize prediction system with proper app context
    with app.app_context():
        try:
            from app.ml.prediction_blueprint import init_prediction_system
            init_prediction_system(app)
        except Exception as e:
            app.logger.warning(f"ML service initialization warning: {e}")
            # Application will continue to work in demo mode
            
        # Initialize recommendation engine
        try:
            from app.ml.recommendation_engine import recommendation_engine
            recommendation_engine.initialize()
            app.logger.info("Recommendation engine initialized successfully")
        except Exception as e:
            app.logger.warning(f"Recommendation engine initialization warning: {e}")
            
    # Register error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('500.html'), 500

    
    import os
    static_folder = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
        
        css_folder = os.path.join(static_folder, 'css')
        if not os.path.exists(css_folder):
            os.makedirs(css_folder)
        
        js_folder = os.path.join(static_folder, 'js')
        if not os.path.exists(js_folder):
            os.makedirs(js_folder)

    return app

from app import models 