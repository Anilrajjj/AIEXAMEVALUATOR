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
from utils.scoring import calculate_score

# Setup logging
logger = logging.getLogger(__name__)

exam_bp = Blueprint('exam', __name__)

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
            keywords=question_form.keywords.data,
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
            
            # If manual entry is selected, skip OCR processing
            if form.use_manual_entry.data:
                # Create submission record for manual entry
                submission = Submission(
                    exam_id=exam_id,
                    student_name=form.student_name.data,
                    student_id=form.student_id.data,
                    original_file=os.path.basename(file_path)
                )
                
                db.session.add(submission)
                db.session.commit()
                
                # Redirect to manual entry form
                flash('File uploaded successfully. Please enter the answers manually.', 'success')
                return redirect(url_for('exam.manual_entry', exam_id=exam_id, submission_id=submission.id))
            
            # Process the file (OCR) with proper error handling and handwritten text detection
            try:
                # Get the handwritten flag from the form (default is True if not specified)
                is_handwritten = form.is_handwritten.data if hasattr(form, 'is_handwritten') else True
                current_app.logger.info(f"Processing file with handwritten mode: {is_handwritten}")
                
                # Process the file with the specified handwritten mode
                extracted_text = process_file(file_path, is_handwritten=is_handwritten)
                current_app.logger.debug(f"Text extraction complete. Text length: {len(extracted_text)}")
                
                # If we got very little text, try the opposite mode as a fallback
                if len(extracted_text.strip()) < 50 and not extracted_text.startswith('[Error'):
                    opposite_mode = not is_handwritten
                    current_app.logger.info(f"Initial extraction yielded little text. Trying {'' if opposite_mode else 'non-'}handwritten mode.")
                    extracted_text = process_file(file_path, is_handwritten=opposite_mode)
                    current_app.logger.debug(f"Fallback text extraction complete. Text length: {len(extracted_text)}")
                
                if not extracted_text or extracted_text.startswith('[Error'):
                    error_msg = "Could not extract text from the uploaded file."
                    if extracted_text.startswith('[Error'):
                        error_msg += f" Details: {extracted_text}"
                    flash(error_msg + " You may want to try manual text entry instead.", 'danger')
                    return render_template('upload_answers.html', form=form, exam=exam)
            except Exception as e:
                current_app.logger.error(f"Exception in text extraction: {str(e)}")
                flash(f'Error processing file: {str(e)}. You may want to try manual text entry instead.', 'danger')
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
            
            # Create submission record
            submission = Submission(
                exam_id=exam_id,
                student_name=student_name,
                student_id=student_id,
                original_file=os.path.basename(file_path)
            )
            
            db.session.add(submission)
            db.session.flush()  # Get submission ID without committing
            
            # Try to segment the extracted text into answers for each question
            text_segments = extract_text_segments(extracted_text, len(questions))
            
            total_score = 0
            
            # Evaluate each answer segment
            for i, (question, text_segment) in enumerate(zip(questions, text_segments)):
                # Evaluate the answer
                similarity_score, keyword_score, feedback = evaluate_answer(
                    text_segment, 
                    question.answer_key,
                    question.keywords,
                    question.question_type
                )
                
                # Calculate final score
                score = calculate_score(similarity_score, keyword_score, question.max_score)
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
            
            db.session.commit()
            
            flash('Answer paper processed successfully!', 'success')
            return redirect(url_for('exam.view_submission', exam_id=exam_id, submission_id=submission.id))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing answer paper: {e}")
            flash(f'Error processing answer paper: {str(e)}', 'danger')
    
    # Get existing submissions for this exam
    submissions = Submission.query.filter_by(exam_id=exam_id).order_by(Submission.submitted_at.desc()).all()
    
    return render_template('upload_answers.html', form=form, exam=exam, submissions=submissions)

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
                
                # Evaluate the answer
                similarity_score, keyword_score, feedback = evaluate_answer(
                    answer_text, 
                    question.answer_key,
                    question.keywords,
                    question.question_type
                )
                
                # Calculate final score
                score = calculate_score(similarity_score, keyword_score, question.max_score)
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
    
    return render_template('view_results.html', exam=exam, submissions=submissions, max_score=max_score)

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
    
    return render_template(
        'evaluate.html', 
        exam=exam, 
        submission=submission, 
        answers=answers,
        max_score=max_score
    )

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
