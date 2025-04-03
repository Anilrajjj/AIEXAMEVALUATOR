import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from flask_login import LoginManager


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)
# create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# configure PostgreSQL database 
db_url = os.environ.get("DATABASE_URL", "sqlite:///instance/app.db")
# Handle SQLAlchemy 1.4+ compatibility with postgres:// URLs
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url

# Ensure we have a database URL
if not app.config["SQLALCHEMY_DATABASE_URI"]:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///instance/app.db"
    logging.warning("No DATABASE_URL found. Using SQLite as fallback.")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max-limit

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# Initialize db
db.init_app(app)

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Register blueprints
from auth import auth_bp
app.register_blueprint(auth_bp)

from routes.exam_routes import exam_bp
app.register_blueprint(exam_bp)

from routes.report_routes import report_bp
app.register_blueprint(report_bp)

# Import models and create tables
with app.app_context():
    import models
    db.create_all()
    logging.info("Database tables created")

# Load user for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))
