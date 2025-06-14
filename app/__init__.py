from flask import Flask
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_mail import Mail
from flask_wtf.csrf import CSRFProtect
from config import Config

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