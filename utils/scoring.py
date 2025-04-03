import logging
import numpy as np
from statistics import mean, median, stdev
from collections import Counter

# Setup logging
logger = logging.getLogger(__name__)

def calculate_score(similarity_score, keyword_score, max_score=1.0, weight_similarity=0.7, weight_keyword=0.3):
    """
    Calculate a final score based on similarity and keyword scores
    
    Args:
        similarity_score: Cosine similarity score (0-1)
        keyword_score: Keyword matching score (0-1)
        max_score: Maximum score for the question
        weight_similarity: Weight for similarity score (default 0.7)
        weight_keyword: Weight for keyword score (default 0.3)
    
    Returns:
        final_score: Weighted score scaled to max_score
    """
    # Calculate weighted combined score
    combined_score = (similarity_score * weight_similarity) + (keyword_score * weight_keyword)
    
    # Scale to max_score
    final_score = combined_score * max_score
    
    return round(final_score, 2)

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
