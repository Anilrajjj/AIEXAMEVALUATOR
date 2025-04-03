import logging
from flask import Blueprint, render_template, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from sqlalchemy import func
from app import db
from models import Exam, Question, Submission, SubmissionAnswer
from utils.scoring import get_score_statistics, generate_score_distribution, identify_common_errors

# Setup logging
logger = logging.getLogger(__name__)

report_bp = Blueprint('report', __name__)

@report_bp.route('/exam/<int:exam_id>/reports')
@login_required
def exam_reports(exam_id):
    """View statistical reports for an exam"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to view reports for this exam.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    # Get all submissions for this exam
    submissions = Submission.query.filter_by(exam_id=exam_id).all()
    
    # Get all questions for this exam
    questions = Question.query.filter_by(exam_id=exam_id).order_by(Question.order).all()
    
    # Calculate maximum possible score
    max_score = sum(q.max_score for q in questions)
    
    # If no submissions, show message
    if not submissions:
        flash('No submissions available for reporting.', 'info')
        return render_template('reports.html', exam=exam, has_data=False)
    
    # Get scores for statistical analysis
    scores = [s.total_score for s in submissions]
    
    # Calculate statistics
    stats = get_score_statistics(scores)
    
    # Generate data for score distribution chart
    labels, counts = generate_score_distribution(scores)
    
    # Calculate average scores per question
    question_avgs = []
    for question in questions:
        # Get all answers for this question
        answers = SubmissionAnswer.query.filter_by(question_id=question.id).all()
        if answers:
            avg_score = sum(a.score for a in answers) / len(answers)
            percentage = (avg_score / question.max_score) * 100 if question.max_score > 0 else 0
            question_avgs.append({
                'question': question,
                'avg_score': round(avg_score, 2),
                'percentage': round(percentage, 1)
            })
    
    # Prepare common error analysis
    error_analysis = {}
    for question in questions:
        # Get all answers for this question
        answers = SubmissionAnswer.query.filter_by(question_id=question.id).all()
        if answers:
            # Identify common errors
            student_answers = [a.extracted_text for a in answers]
            answer_keys = [question.answer_key] * len(answers)
            
            # This function will identify common missing keywords
            missing_keywords = identify_common_errors(student_answers, answer_keys)
            
            error_analysis[question.id] = {
                'question': question.text,
                'missing_keywords': missing_keywords
            }
    
    return render_template(
        'reports.html',
        exam=exam,
        has_data=True,
        submissions_count=len(submissions),
        max_score=max_score,
        stats=stats,
        chart_labels=labels,
        chart_data=counts,
        question_avgs=question_avgs,
        error_analysis=error_analysis
    )

@report_bp.route('/api/exam/<int:exam_id>/score_distribution')
@login_required
def score_distribution_data(exam_id):
    """API endpoint to get score distribution data for charts"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        return jsonify({'error': 'Not authorized'}), 403
    
    # Get all submissions for this exam
    submissions = Submission.query.filter_by(exam_id=exam_id).all()
    
    # Get scores
    scores = [s.total_score for s in submissions]
    
    # Generate data for chart
    labels, counts = generate_score_distribution(scores)
    
    return jsonify({
        'labels': labels,
        'counts': counts
    })

@report_bp.route('/api/exam/<int:exam_id>/question_performance')
@login_required
def question_performance_data(exam_id):
    """API endpoint to get question performance data for charts"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        return jsonify({'error': 'Not authorized'}), 403
    
    # Get all questions for this exam
    questions = Question.query.filter_by(exam_id=exam_id).order_by(Question.order).all()
    
    # Calculate average scores per question
    question_data = []
    for question in questions:
        # Get average score for this question
        avg_score = db.session.query(func.avg(SubmissionAnswer.score)) \
                              .filter(SubmissionAnswer.question_id == question.id) \
                              .scalar() or 0
        
        percentage = (avg_score / question.max_score) * 100 if question.max_score > 0 else 0
        
        question_data.append({
            'question': f"Q{question.order + 1}",
            'avg_score': round(avg_score, 2),
            'percentage': round(percentage, 1),
            'max_score': question.max_score
        })
    
    return jsonify({
        'questions': [q['question'] for q in question_data],
        'percentages': [q['percentage'] for q in question_data],
        'avg_scores': [q['avg_score'] for q in question_data],
        'max_scores': [q['max_score'] for q in question_data]
    })
