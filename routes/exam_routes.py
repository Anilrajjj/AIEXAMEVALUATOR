import os
import logging
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, jsonify, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from models import Exam, Question, Submission, SubmissionAnswer
from forms import ExamForm, QuestionForm, SubmissionForm, ManualAnswerForm
from utils.ocr_processor import process_file, save_uploaded_file, extract_text_segments
from utils.nlp_processor import evaluate_answer, extract_student_info
from utils.scoring import (
    calculate_score,
    calculate_grade,
    determine_pass_status,
    get_passing_mark,
    get_score_statistics,
    estimate_language_quality,
    PASSING_PERCENTAGE,
)
from datetime import datetime
import json
from flask import make_response

# Setup logging
logger = logging.getLogger(__name__)

exam_bp = Blueprint('exam', __name__)


def build_submission_report(exam, submission, answers, max_score, evaluator_name):
    scores = [s.total_score for s in exam.submissions if s.total_score is not None]
    stats = get_score_statistics(scores)

    percentage = (submission.total_score / max_score) * 100 if max_score else 0
    grade = calculate_grade(submission.total_score, max_score) if max_score else 'N/A'
    passing_mark, pass_status = determine_pass_status(submission.total_score or 0, max_score)

    question_details = []
    section_map = {}

    for answer in answers:
        grammar_score, spelling_score = estimate_language_quality(answer.extracted_text or "")
        question_details.append({
            'question_number': answer.question.order + 1,
            'question_text': answer.question.text,
            'max_marks': answer.question.max_score,
            'similarity': round(answer.similarity_score or 0, 2),
            'grammar': grammar_score,
            'spelling': spelling_score,
            'final_score': answer.score
        })

        section_key = answer.question.question_type or 'General'
        if section_key not in section_map:
            section_map[section_key] = {
                'section': section_key.title(),
                'total_marks': 0,
                'obtained': 0
            }
        section_map[section_key]['total_marks'] += answer.question.max_score
        section_map[section_key]['obtained'] += answer.score

    section_summary = list(section_map.values())

    feedback_summary = (
        f"{submission.student_name} scored {submission.total_score:.1f} out of {max_score:.1f} "
        f"({percentage:.1f}%). Status: {pass_status}. Focus on questions with low similarity "
        "to improve understanding."
    )

    analytics = {
        'mean': stats['mean'],
        'median': stats['median'],
        'std_dev': stats['std_dev'],
        'score_distribution': [
            {'student': s.student_name, 'score': s.total_score}
            for s in sorted(exam.submissions, key=lambda x: x.total_score or 0, reverse=True)
        ]
    }

    report_data = {
        'student_info': {
            'name': submission.student_name,
            'roll_number': submission.student_id,
            'subject': exam.title,
            'total_marks': max_score
        },
        'question_details': question_details,
        'section_summary': section_summary,
        'overall_result': {
            'total_score': submission.total_score,
            'percentage': round(percentage, 2),
            'grade': grade,
            'passing_mark': passing_mark,
            'status': pass_status
        },
        'analytics': analytics,
        'feedback_summary': feedback_summary,
        'metadata': {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'evaluator': evaluator_name
        }
    }

    return report_data

@exam_bp.route('/dashboard')
@login_required
def dashboard():
    """Display faculty dashboard with list of exams"""
    exams = Exam.query.filter_by(faculty_id=current_user.id).order_by(Exam.created_at.desc()).all()
    return render_template('dashboard.html', exams=exams)

@exam_bp.route('/exam/create', methods=['GET', 'POST'])
@login_required
def create_exam():
    """Create a new exam"""
    form = ExamForm()
    
    if form.validate_on_submit():
        exam = Exam(
            title=form.title.data,
            description=form.description.data,
            faculty_id=current_user.id
        )
        
        db.session.add(exam)
        db.session.commit()
        
        flash('Exam created successfully!', 'success')
        return redirect(url_for('exam.edit_exam', exam_id=exam.id))
    
    return render_template('upload_exam.html', form=form, title="Create New Exam")

@exam_bp.route('/exam/<int:exam_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_exam(exam_id):
    """Edit an existing exam and manage questions"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to edit this exam.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    # Get all questions for this exam, ordered by their order field
    questions = Question.query.filter_by(exam_id=exam_id).order_by(Question.order).all()
    
    # Handle the exam update form
    exam_form = ExamForm(obj=exam)
    if exam_form.validate_on_submit() and 'update_exam' in request.form:
        exam_form.populate_obj(exam)
        db.session.commit()
        flash('Exam updated successfully!', 'success')
        return redirect(url_for('exam.edit_exam', exam_id=exam_id))
    
    # Handle the question form
    question_form = QuestionForm()
    if question_form.validate_on_submit() and 'add_question' in request.form:
        question = Question(
            exam_id=exam_id,
            text=question_form.text.data,
            answer_key=question_form.answer_key.data,
            min_word_count=question_form.min_word_count.data,
            max_score=question_form.max_score.data,
            question_type=question_form.question_type.data,
            order=question_form.order.data
        )
        
        db.session.add(question)
        db.session.commit()
        
        flash('Question added successfully!', 'success')
        return redirect(url_for('exam.edit_exam', exam_id=exam_id))
    
    return render_template(
        'upload_exam.html', 
        exam=exam, 
        exam_form=exam_form, 
        question_form=question_form,
        questions=questions,
        title="Edit Exam"
    )

@exam_bp.route('/exam/<int:exam_id>/question/<int:question_id>/delete', methods=['POST'])
@login_required
def delete_question(exam_id, question_id):
    """Delete a question from an exam"""
    question = Question.query.get_or_404(question_id)
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to edit this exam.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    db.session.delete(question)
    db.session.commit()
    
    flash('Question deleted successfully!', 'success')
    return redirect(url_for('exam.edit_exam', exam_id=exam_id))

@exam_bp.route('/exam/<int:exam_id>/question/<int:question_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_question(exam_id, question_id):
    """Edit a question"""
    question = Question.query.get_or_404(question_id)
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to edit this exam.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    form = QuestionForm(obj=question)
    
    if form.validate_on_submit():
        form.populate_obj(question)
        db.session.commit()
        
        flash('Question updated successfully!', 'success')
        return redirect(url_for('exam.edit_exam', exam_id=exam_id))
    
    return render_template(
        'upload_exam.html', 
        exam=exam, 
        question=question, 
        question_form=form,
        title="Edit Question"
    )

@exam_bp.route('/exam/<int:exam_id>/upload', methods=['GET', 'POST'])
@login_required
def upload_answers(exam_id):
    """Upload and process student answer papers"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to upload answers for this exam.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    # Get the questions for this exam
    questions = Question.query.filter_by(exam_id=exam_id).order_by(Question.order).all()
    
    if not questions:
        flash('Please add questions to this exam before uploading answer papers.', 'warning')
        return redirect(url_for('exam.edit_exam', exam_id=exam_id))
    
    form = SubmissionForm()
    
    if form.validate_on_submit():
        try:
            # Save the uploaded file
            upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], f'exam_{exam_id}')
            os.makedirs(upload_folder, exist_ok=True)
            
            # Get the file and validate its type
            upload_file = form.answer_file.data
            filename = upload_file.filename.lower()
            if not (filename.endswith('.pdf') or 
                    filename.endswith('.jpg') or 
                    filename.endswith('.jpeg') or 
                    filename.endswith('.png')):
                flash('Please upload a PDF or image file (jpg, jpeg, png).', 'danger')
                return render_template('upload_answers.html', form=form, exam=exam)
            
            # Log the upload
            current_app.logger.info(f"Processing upload: {filename} for exam {exam_id}")
            
            # Save the file
            file_path = save_uploaded_file(upload_file, upload_folder)
            current_app.logger.debug(f"File saved at: {file_path}")
            
            # Process the file (OCR) with automatic detection
            try:
                # Always try handwritten mode first for better accuracy
                current_app.logger.info("Processing file with automatic OCR detection")
                
                # Process the file with handwritten mode enabled
                extracted_text = process_file(file_path, is_handwritten=True)
                current_app.logger.debug(f"Text extraction complete. Text length: {len(extracted_text)}")
                # Debug preview of extracted text to help diagnose OCR/segmentation issues
                preview = (extracted_text[:1000] + '...') if len(extracted_text) > 1000 else extracted_text
                current_app.logger.debug(f"EXTRACTED_TEXT_PREVIEW: {preview!r}")
                
                # If we got very little text, try printed text mode as a fallback
                if len(extracted_text.strip()) < 50 and not extracted_text.startswith('[Error'):
                    current_app.logger.info("Initial extraction yielded little text. Trying printed text mode.")
                    extracted_text = process_file(file_path, is_handwritten=False)
                    current_app.logger.debug(f"Fallback text extraction complete. Text length: {len(extracted_text)}")
                
                if not extracted_text or extracted_text.startswith('[Error'):
                    error_msg = "Could not extract text from the uploaded file."
                    if extracted_text.startswith('[Error'):
                        error_msg += f" Details: {extracted_text}"
                    flash(error_msg + " Please try uploading a clearer image or PDF.", 'danger')
                    return render_template('upload_answers.html', form=form, exam=exam)
            except Exception as e:
                current_app.logger.error(f"Exception in text extraction: {str(e)}")
                flash(f'Error processing file: {str(e)}. Please try uploading a clearer image or PDF.', 'danger')
                return render_template('upload_answers.html', form=form, exam=exam)
            
            # Try to extract student info from the text
            try:
                auto_student_name, auto_student_id = extract_student_info(extracted_text)
                current_app.logger.debug(f"Extracted student info: {auto_student_name}, {auto_student_id}")
            except Exception as e:
                current_app.logger.error(f"Error extracting student info: {str(e)}")
                auto_student_name, auto_student_id = "", ""
            
            # Use form data but fall back to extracted info if not provided
            student_name = form.student_name.data or auto_student_name
            student_id = form.student_id.data or auto_student_id
            
            # Create submission record with proper transaction handling
            try:
                submission = Submission(
                    exam_id=exam_id,
                    student_name=student_name,
                    student_id=student_id,
                    original_file=os.path.basename(file_path),
                    extracted_text=extracted_text  # Store full extracted text in DB
                )
                
                db.session.add(submission)
                db.session.flush()  # Get submission ID without committing
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error creating submission record: {e}")
                flash(f'Error creating submission: {str(e)}', 'danger')
                return render_template('upload_answers.html', form=form, exam=exam)
            
            # Try to segment the extracted text into answers for each question
            text_segments = extract_text_segments(extracted_text, len(questions))

            # Align extracted segments to questions using TF-IDF if segmentation order is unreliable
            try:
                from utils.aligner import align_segments_to_questions
                question_texts = [q.text for q in questions]
                aligned_segments = align_segments_to_questions(question_texts, text_segments)
                # If aligner returned empty segments for all questions, fall back to original order
                if not any(s.strip() for s in aligned_segments):
                    aligned_segments = text_segments
            except Exception:
                aligned_segments = text_segments
            
            total_score = 0
            
            # Evaluate each answer segment (use aligned_segments)
            for i, (question, text_segment) in enumerate(zip(questions, aligned_segments)):
                # Evaluate the answer (returns: relevance, word_length, feedback, final_marks)
                similarity_score, keyword_score, feedback, final_marks = evaluate_answer(
                    text_segment, 
                    question.answer_key,
                    question.text,
                    question.max_score,
                    question.min_word_count
                )
                
                # Calculate final score using 4-parameter system
                score = calculate_score(similarity_score, keyword_score, question.max_score, text_segment, final_marks)
                total_score += score
                
                # Create submission answer record
                submission_answer = SubmissionAnswer(
                    submission_id=submission.id,
                    question_id=question.id,
                    extracted_text=text_segment,
                    score=score,
                    similarity_score=similarity_score,
                    feedback=feedback
                )
                
                db.session.add(submission_answer)
            
            # Update total score
            submission.total_score = total_score
            
            # Commit with retry logic for database locks
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    db.session.commit()
                    break
                except Exception as commit_error:
                    if attempt < max_retries - 1:
                        logger.warning(f"Database commit attempt {attempt + 1} failed: {commit_error}. Retrying...")
                        db.session.rollback()
                        import time
                        time.sleep(0.5)  # Wait before retry
                    else:
                        logger.error(f"All database commit attempts failed: {commit_error}")
                        db.session.rollback()
                        flash(f'Error saving submission: {str(commit_error)}', 'danger')
                        return render_template('upload_answers.html', form=form, exam=exam)
            
            flash('Answer paper processed successfully!', 'success')
            # Store extracted text in session for review
            session['extracted_text'] = extracted_text
            session['submission_id'] = submission.id
            return redirect(url_for('exam.review_extraction', exam_id=exam_id, submission_id=submission.id))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing answer paper: {e}")
            flash(f'Error processing answer paper: {str(e)}', 'danger')
    
    # Get existing submissions for this exam
    submissions = Submission.query.filter_by(exam_id=exam_id).order_by(Submission.submitted_at.desc()).all()
    
    return render_template('upload_answers.html', form=form, exam=exam, submissions=submissions)

@exam_bp.route('/exam/<int:exam_id>/submission/<int:submission_id>/review')
@login_required
def review_extraction(exam_id, submission_id):
    """Review extracted text before grading"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to view this submission.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    submission = Submission.query.get_or_404(submission_id)
    
    # Ensure the submission belongs to the specified exam
    if submission.exam_id != exam_id:
        flash('Submission does not belong to the specified exam.', 'danger')
        return redirect(url_for('exam.view_submissions', exam_id=exam_id))
    
    # Get extracted text from the submission DB record (not session)
    extracted_text = submission.extracted_text or ''
    
    if not extracted_text:
        flash('No extracted text found. Processing may have failed.', 'warning')
        return redirect(url_for('exam.view_submission', exam_id=exam_id, submission_id=submission_id))
    
    # Clean the extracted text by removing processing headers and metadata
    cleaned_text = extracted_text
    
    # Remove TrOCR header
    if cleaned_text.startswith('[TrOCR'):
        # Find the end of the header (look for the first double newline or "---")
        lines = cleaned_text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('[') and line.strip().endswith(']'):
                start_idx = i + 1
            elif line.strip() == '' or line.startswith('---'):
                start_idx = i + 1
            else:
                break
        cleaned_text = '\n'.join(lines[start_idx:]).strip()
    
    # Remove other processing metadata lines
    lines = cleaned_text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that are metadata/headers
        if not line.strip().startswith('[') or line.strip().startswith('--- Page'):
            cleaned_lines.append(line)
    
    cleaned_text = '\n'.join(cleaned_lines).strip()
    
    return render_template(
        'review_extraction.html', 
        exam=exam, 
        submission=submission, 
        extracted_text=extracted_text,
        cleaned_text=cleaned_text
    )

@exam_bp.route('/exam/<int:exam_id>/submission/<int:submission_id>/manual-entry', methods=['GET', 'POST'])
@login_required
def manual_entry(exam_id, submission_id):
    """Manually enter student answers for a submission"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to edit this submission.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    submission = Submission.query.get_or_404(submission_id)
    
    # Ensure the submission belongs to the specified exam
    if submission.exam_id != exam_id:
        flash('Submission does not belong to the specified exam.', 'danger')
        return redirect(url_for('exam.view_submissions', exam_id=exam_id))
    
    # Get the questions for this exam
    questions = Question.query.filter_by(exam_id=exam_id).order_by(Question.order).all()
    
    # Check if this submission already has answers
    existing_answers = SubmissionAnswer.query.filter_by(submission_id=submission_id).all()
    
    # If there are existing answers, redirect to view submission
    if existing_answers and request.method == 'GET':
        flash('This submission already has answers. You can view them below.', 'info')
        return redirect(url_for('exam.view_submission', exam_id=exam_id, submission_id=submission_id))
    
    # If form is submitted (POST request)
    if request.method == 'POST':
        try:
            total_score = 0
            
            # Process each question's manual answer
            for question in questions:
                answer_text = request.form.get(f'answer_{question.id}')
                
                if not answer_text:
                    answer_text = "No answer provided."
                
                # Evaluate the answer (returns: relevance, word_length, feedback, final_marks)
                similarity_score, keyword_score, feedback, final_marks = evaluate_answer(
                    answer_text, 
                    question.answer_key,
                    question.text,
                    question.max_score,
                    question.min_word_count
                )
                
                # Calculate final score using 4-parameter system
                score = calculate_score(similarity_score, keyword_score, question.max_score, answer_text, final_marks)
                total_score += score
                
                # Create or update submission answer record
                existing_answer = SubmissionAnswer.query.filter_by(
                    submission_id=submission_id, 
                    question_id=question.id
                ).first()
                
                if existing_answer:
                    existing_answer.extracted_text = answer_text
                    existing_answer.score = score
                    existing_answer.similarity_score = similarity_score
                    existing_answer.feedback = feedback
                else:
                    submission_answer = SubmissionAnswer(
                        submission_id=submission_id,
                        question_id=question.id,
                        extracted_text=answer_text,
                        score=score,
                        similarity_score=similarity_score,
                        feedback=feedback
                    )
                    db.session.add(submission_answer)
            
            # Update total score
            submission.total_score = total_score
            db.session.commit()
            
            flash('Answers saved and evaluated successfully!', 'success')
            return redirect(url_for('exam.view_submission', exam_id=exam_id, submission_id=submission_id))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing manual answers: {e}")
            flash(f'Error processing answers: {str(e)}', 'danger')
    
    # For GET request, prepare the form for manual entry
    return render_template(
        'manual_entry.html', 
        exam=exam, 
        submission=submission, 
        questions=questions
    )

@exam_bp.route('/exam/<int:exam_id>/submissions')
@login_required
def view_submissions(exam_id):
    """View all submissions for an exam"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to view submissions for this exam.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    # Get all submissions for this exam
    submissions = Submission.query.filter_by(exam_id=exam_id).order_by(Submission.submitted_at.desc()).all()
    
    # Calculate maximum possible score for the exam
    questions = Question.query.filter_by(exam_id=exam_id).all()
    max_score = sum(q.max_score for q in questions)
    
    passing_mark = get_passing_mark(max_score)

    submission_rows = []
    for submission in submissions:
        _, pass_status = determine_pass_status(submission.total_score or 0, max_score)
        submission_rows.append({
            'submission': submission,
            'pass_status': pass_status
        })

    return render_template(
        'view_results.html',
        exam=exam,
        submissions=submissions,
        submission_rows=submission_rows,
        max_score=max_score,
        passing_mark=passing_mark,
        passing_percentage=int(PASSING_PERCENTAGE * 100)
    )

@exam_bp.route('/exam/<int:exam_id>/submission/<int:submission_id>')
@login_required
def view_submission(exam_id, submission_id):
    """View details of a specific submission"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to view this submission.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    submission = Submission.query.get_or_404(submission_id)
    
    # Ensure the submission belongs to the specified exam
    if submission.exam_id != exam_id:
        flash('Submission does not belong to the specified exam.', 'danger')
        return redirect(url_for('exam.view_submissions', exam_id=exam_id))
    
    # Get answers with their corresponding questions
    answers = SubmissionAnswer.query.filter_by(submission_id=submission_id).all()
    
    # Get maximum possible score
    questions = Question.query.filter_by(exam_id=exam_id).all()
    max_score = sum(q.max_score for q in questions)
    
    report_data = build_submission_report(
        exam,
        submission,
        answers,
        max_score,
        current_user.username
    )
    
    return render_template(
        'evaluate.html', 
        exam=exam, 
        submission=submission, 
        answers=answers,
        max_score=max_score,
        report_data=report_data
    )


@exam_bp.route('/exam/<int:exam_id>/submission/<int:submission_id>/report.<string:export_format>')
@login_required
def download_submission_report(exam_id, submission_id, export_format):
    """Download detailed report for a submission."""
    exam = Exam.query.get_or_404(exam_id)
    
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to download this report.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    submission = Submission.query.get_or_404(submission_id)
    if submission.exam_id != exam_id:
        flash('Submission does not belong to the specified exam.', 'danger')
        return redirect(url_for('exam.view_submissions', exam_id=exam_id))
    
    answers = SubmissionAnswer.query.filter_by(submission_id=submission_id).all()
    questions = Question.query.filter_by(exam_id=exam_id).all()
    max_score = sum(q.max_score for q in questions)
    
    report_data = build_submission_report(
        exam,
        submission,
        answers,
        max_score,
        current_user.username
    )
    
    if export_format.lower() == 'json':
        response = make_response(json.dumps(report_data, indent=2))
        response.mimetype = 'application/json'
        filename = f"{submission.student_id}_report.json"
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    
    flash('Only JSON export is supported currently.', 'warning')
    return redirect(url_for('exam.view_submission', exam_id=exam_id, submission_id=submission_id))

@exam_bp.route('/exam/<int:exam_id>/delete', methods=['POST'])
@login_required
def delete_exam(exam_id):
    """Delete an exam and all associated data"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to delete this exam.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    # Delete the exam (cascade will handle related records)
    db.session.delete(exam)
    db.session.commit()
    
    flash('Exam deleted successfully!', 'success')
    return redirect(url_for('exam.dashboard'))

@exam_bp.route('/exam/<int:exam_id>/submission/<int:submission_id>/delete', methods=['POST'])
@login_required
def delete_submission(exam_id, submission_id):
    """Delete a submission"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Ensure the current user owns this exam
    if exam.faculty_id != current_user.id:
        flash('You do not have permission to delete this submission.', 'danger')
        return redirect(url_for('exam.dashboard'))
    
    submission = Submission.query.get_or_404(submission_id)
    
    # Ensure the submission belongs to the specified exam
    if submission.exam_id != exam_id:
        flash('Submission does not belong to the specified exam.', 'danger')
        return redirect(url_for('exam.view_submissions', exam_id=exam_id))
    
    # Delete the submission (cascade will handle related records)
    db.session.delete(submission)
    db.session.commit()
    
    flash('Submission deleted successfully!', 'success')
    return redirect(url_for('exam.view_submissions', exam_id=exam_id))
