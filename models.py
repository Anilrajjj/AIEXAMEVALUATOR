from datetime import datetime
from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    is_faculty = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    exams = db.relationship('Exam', backref='faculty', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    faculty_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    questions = db.relationship('Question', backref='exam', lazy=True, cascade="all, delete-orphan")
    submissions = db.relationship('Submission', backref='exam', lazy=True, cascade="all, delete-orphan")


class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    answer_key = db.Column(db.Text, nullable=False)
    max_score = db.Column(db.Float, default=1.0)
    question_type = db.Column(db.String(20), default='text')  # text, mcq, etc.
    keywords = db.Column(db.Text)  # Comma-separated keywords for evaluation
    order = db.Column(db.Integer, default=0)


class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    student_name = db.Column(db.String(128), nullable=False)
    student_id = db.Column(db.String(64), nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    original_file = db.Column(db.String(255))  # Path to uploaded file
    total_score = db.Column(db.Float, default=0)
    answers = db.relationship('SubmissionAnswer', backref='submission', lazy=True, cascade="all, delete-orphan")


class SubmissionAnswer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id'), nullable=False)
    extracted_text = db.Column(db.Text)
    score = db.Column(db.Float, default=0)
    similarity_score = db.Column(db.Float)
    feedback = db.Column(db.Text)
    
    question = db.relationship('Question')
