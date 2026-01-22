import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from flask_login import LoginManager


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)
# create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True

# Configure SQLite database
db_path = os.path.join(os.getcwd(), 'instance', 'app.db')
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}?timeout=30"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "pool_timeout": 30,
    "max_overflow": 10,
    "pool_size": 5,
    "connect_args": {
        "timeout": 30,
        "check_same_thread": False
    }
}

# Ensure instance directory exists with proper permissions
instance_path = os.path.join(os.getcwd(), 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path, mode=0o777)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max-limit

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# Initialize db
db.init_app(app)

# Create required directories
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("instance", exist_ok=True)

# Initialize db tables in app context
with app.app_context():
    db.create_all()

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
    try:
        # Try to convert to int (SQLite integer ID)
        return User.query.get(int(user_id))
    except (ValueError, TypeError):
        # If conversion fails (e.g., MongoDB ObjectId string), return None
        return None

# Add database cleanup on app teardown
@app.teardown_appcontext
def close_db(error):
    """Close database connections properly"""
    try:
        if hasattr(db, 'session'):
            db.session.remove()
    except Exception as e:
        logging.warning(f"Error closing database session: {e}")
