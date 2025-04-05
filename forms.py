from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, BooleanField, TextAreaField, FloatField, IntegerField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError, Optional

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(), 
        EqualTo('password', message='Passwords must match')
    ])

class ExamForm(FlaskForm):
    title = StringField('Exam Title', validators=[DataRequired(), Length(max=128)])
    description = TextAreaField('Description', validators=[Optional(), Length(max=500)])

class QuestionForm(FlaskForm):
    text = TextAreaField('Question', validators=[DataRequired()])
    answer_key = TextAreaField('Answer Key', validators=[DataRequired()])
    keywords = TextAreaField('Keywords (comma separated)', validators=[Optional()])
    max_score = FloatField('Maximum Score', validators=[DataRequired()], default=1.0)
    question_type = SelectField('Question Type', choices=[
        ('text', 'Text Answer'),
        ('mcq', 'Multiple Choice'),
        ('short', 'Short Answer')
    ], default='text')
    order = IntegerField('Order', default=0)

class SubmissionForm(FlaskForm):
    student_name = StringField('Student Name', validators=[DataRequired(), Length(max=128)])
    student_id = StringField('Student ID', validators=[DataRequired(), Length(max=64)])
    answer_file = FileField('Answer Paper (PDF/Image)', validators=[
        FileRequired(),
        FileAllowed(['pdf', 'jpg', 'jpeg', 'png'], 'Only PDF and image files are allowed!')
    ])
    is_handwritten = BooleanField('This is a handwritten answer paper', default=True)
    use_manual_entry = BooleanField('Use manual text entry (in case OCR fails)', default=False)

class ManualAnswerForm(FlaskForm):
    answer_text = TextAreaField('Student Answer Text', validators=[DataRequired()])
    question_id = IntegerField('Question ID', validators=[DataRequired()])
