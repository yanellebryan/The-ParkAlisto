from flask import Flask
from config import Config
from config.logging_config import setup_logging

def create_app(config_class=Config):
    setup_logging()
    
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    app.config.from_object(config_class)

    # Register Blueprints
    from app.routes import api_bp, web_bp
    app.register_blueprint(api_bp)
    app.register_blueprint(web_bp)

    return app
