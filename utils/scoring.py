import logging
from math import ceil
import re
import numpy as np
from statistics import mean, median, stdev
from collections import Counter

# Setup logging
logger = logging.getLogger(__name__)

PASSING_PERCENTAGE = 0.35

PASSING_REFERENCE_TABLE = [
    {"total_marks": 20, "raw": 7.0, "passing_mark": 7},
    {"total_marks": 25, "raw": 8.75, "passing_mark": 9},
    {"total_marks": 30, "raw": 10.5, "passing_mark": 11},
    {"total_marks": 35, "raw": 12.25, "passing_mark": 13},
    {"total_marks": 40, "raw": 14.0, "passing_mark": 14},
    {"total_marks": 50, "raw": 17.5, "passing_mark": 18},
    {"total_marks": 60, "raw": 21.0, "passing_mark": 21},
    {"total_marks": 70, "raw": 24.5, "passing_mark": 25},
    {"total_marks": 80, "raw": 28.0, "passing_mark": 28},
    {"total_marks": 90, "raw": 31.5, "passing_mark": 32},
    {"total_marks": 100, "raw": 35.0, "passing_mark": 35},
]

def calculate_score(similarity_score, keyword_score, max_score=1.0, answer_text="", final_marks=None):
    """
    Calculate score using the new 4-parameter grading system or fallback.
    
    Args:
        similarity_score: Relevance score (0-1) - from new system or legacy
        keyword_score: Word length score (0-1) - from new system or legacy keyword matching
        max_score: Maximum score for the question
        answer_text: Student's answer text
        final_marks: Final marks from new grading system (if available)
    
    Returns:
        final_score: Score based on 4-parameter system or fallback
    """
    # If new grading system provided final marks, use it directly
    if final_marks is not None:
        return min(max_score, max(0, round(final_marks, 2)))
    
    # Fallback to legacy scoring for backward compatibility
    logger.info("Using legacy scoring fallback")
    
    # Get answer length
    answer_length = len(answer_text.split())
    
    # Base score from keyword matching
    base_score = keyword_score * max_score
    
    # Length-based scoring rules
    if max_score <= 2:
        # For 1-2 mark questions
        min_words = 10
        ideal_words = 30
    else:
        # For questions worth more than 2 marks
        min_words = 50
        ideal_words = 150
    
    # Adjust score based on length
    if answer_length < min_words:
        length_factor = 0.5
    elif answer_length >= ideal_words:
        length_factor = 1.0
    else:
        length_factor = 0.5 + (0.5 * (answer_length - min_words) / (ideal_words - min_words))
    
    # Consider similarity for similar meaning matches
    meaning_bonus = similarity_score * 0.3 * max_score
    
    # Calculate final score
    final_score = (base_score * length_factor) + meaning_bonus
    
    # Ensure score is within bounds
    final_score = min(max_score, max(0, round(final_score, 2)))
    
    return final_score

def get_passing_mark(total_marks, passing_percentage=PASSING_PERCENTAGE):
    """
    Calculate the minimum passing mark based on a percentage of total marks.

    Args:
        total_marks (float): Total obtainable marks.
        passing_percentage (float): Percentage required to pass (default 35%).

    Returns:
        int: Passing mark rounded up to the nearest whole number.
    """
    if total_marks is None or total_marks <= 0:
        return 0

    return int(ceil(total_marks * passing_percentage))


def determine_pass_status(score, total_marks, passing_percentage=PASSING_PERCENTAGE):
    """
    Determine pass/fail status for a student score.

    Args:
        score (float): Student's obtained score.
        total_marks (float): Total obtainable marks.
        passing_percentage (float): Percentage required to pass.

    Returns:
        tuple: (passing_mark, status) where status is 'Pass' or 'Fail'.
    """
    passing_mark = get_passing_mark(total_marks, passing_percentage)
    status = "Pass" if score >= passing_mark else "Fail"

    return passing_mark, status


def estimate_language_quality(text):
    """
    Lightweight heuristic for grammar and spelling quality scores.

    Returns:
        tuple: (grammar_score, spelling_score) in range 0-1
    """
    if not text or not text.strip():
        return 0.0, 0.0

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)

    avg_sentence_length = len(words) / max(len(sentences), 1)
    grammar_score = min(1.0, avg_sentence_length / 25.0)

    alphabetic_words = [w for w in words if w.isalpha()]
    clean_words = [w for w in alphabetic_words if len(w) > 2]
    proper_case_words = [w for w in clean_words if w[0].isupper()]

    if clean_words:
        spelling_score = 1.0 - (len(proper_case_words) / (len(clean_words) * 3))
        spelling_score = max(0.0, min(1.0, spelling_score))
    else:
        spelling_score = 0.0

    return round(grammar_score, 2), round(spelling_score, 2)

def get_score_statistics(scores):
    """
    Calculate statistics for a list of scores
    
    Args:
        scores: List of numeric scores
    
    Returns:
        dict: Statistics including min, max, mean, median, and standard deviation
    """
    if not scores:
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'std_dev': 0,
            'count': 0
        }
    
    try:
        stats = {
            'min': min(scores),
            'max': max(scores),
            'mean': mean(scores),
            'median': median(scores),
            'std_dev': stdev(scores) if len(scores) > 1 else 0,
            'count': len(scores)
        }
        return stats
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'std_dev': 0,
            'count': len(scores)
        }

def calculate_grade(score, max_score):
    """
    Calculate letter grade based on percentage
    
    Args:
        score: Numeric score
        max_score: Maximum possible score
    
    Returns:
        str: Letter grade
    """
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    
    if percentage >= 90:
        return 'A'
    elif percentage >= 80:
        return 'B'
    elif percentage >= 70:
        return 'C'
    elif percentage >= 60:
        return 'D'
    else:
        return 'F'

def generate_score_distribution(scores, num_bins=10):
    """
    Generate a histogram data for score distribution
    
    Args:
        scores: List of numeric scores
        num_bins: Number of bins for histogram
    
    Returns:
        tuple: (bin_labels, bin_counts) for chart rendering
    """
    if not scores:
        return [], []
    
    try:
        min_score = min(scores)
        max_score = max(scores)
        
        # If all scores are the same, create a single bin
        if min_score == max_score:
            return [f"{min_score:.1f}"], [len(scores)]
        
        # Create bins and count scores in each bin
        bin_width = (max_score - min_score) / num_bins
        bins = [min_score + i * bin_width for i in range(num_bins + 1)]
        
        hist, bin_edges = np.histogram(scores, bins=bins)
        
        # Create labels for the bins
        bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
        
        return bin_labels, hist.tolist()
    
    except Exception as e:
        logger.error(f"Error generating score distribution: {e}")
        return [], []

def identify_common_errors(student_answers, answer_keys):
    """
    Identify common missing keywords across all submissions
    
    Args:
        student_answers: List of student answer texts
        answer_keys: List of answer key texts
    
    Returns:
        dict: Common missing keywords and their frequencies
    """
    all_missing_keywords = []
    
    for student_answer, answer_key in zip(student_answers, answer_keys):
        _, missing, _ = keyword_matching(student_answer, answer_key)
        all_missing_keywords.extend(missing)
    
    # Count frequencies of missing keywords
    keyword_counter = Counter(all_missing_keywords)
    
    # Return the most common missing keywords
    return dict(keyword_counter.most_common(10))

def keyword_matching(student_answer, answer_key):
    """Simple keyword matching for identifying common errors"""
    # Extract keywords from answer key (simple version)
    keywords = [word.lower() for word in answer_key.split() if len(word) > 3]
    
    # Check which keywords are missing in student answer
    student_lower = student_answer.lower()
    
    matching = []
    missing = []
    
    for keyword in keywords:
        if keyword in student_lower:
            matching.append(keyword)
        else:
            missing.append(keyword)
    
    score = len(matching) / len(keywords) if keywords else 0
    
    return matching, missing, score
