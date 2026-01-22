import logging
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")

# Initialize tokenizer with fallback
try:
    from nltk.tokenize import word_tokenize
    tokenizer = word_tokenize
    logger.info("Using NLTK word tokenizer")
except Exception as e:
    logger.warning(f"Failed to load NLTK tokenizer: {e}. Using simple split.")
    tokenizer = lambda text: text.split()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text for NLP analysis with basic tokenization
    """
    if not text:
        return ""
        
    # Clean any NULL bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Simple word tokenization
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(processed_tokens)

def keyword_matching(student_answer, answer_key, keywords=None):
    """
    Check for the presence of keywords in the student's answer
    Returns a tuple of (matching_keywords, missing_keywords, score)
    """
    if not keywords:
        # Extract potential keywords from answer key (non-stopwords)
        try:
            answer_tokens = tokenizer(answer_key.lower())
        except:
            answer_tokens = answer_key.lower().split()
        keywords = [token for token in answer_tokens if token not in stop_words and len(token) > 3]
    else:
        # Use provided keywords, split by comma and strip whitespace
        keywords = [k.strip().lower() for k in keywords.split(',')]

    student_answer = student_answer.lower()

    # Check for each keyword in the student answer
    matching_keywords = []
    missing_keywords = []

    for keyword in keywords:
        if keyword in student_answer:
            matching_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    # Calculate a simple score based on the proportion of matching keywords
    if len(keywords) > 0:
        score = len(matching_keywords) / len(keywords)
    else:
        score = 0

    return (matching_keywords, missing_keywords, score)

def calculate_relevance(student_answer, answer_key):
    """
    Calculate relevance score using TF-IDF cosine similarity.
    Returns a value between 0 and 1.
    """
    try:
        # Clean inputs
        student_answer = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', student_answer)
        answer_key = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', answer_key)
        
        if not student_answer.strip() or not answer_key.strip():
            return 0.0
        
        # Preprocess texts
        processed_student = preprocess_text(student_answer)
        processed_key = preprocess_text(answer_key)
        
        if not processed_student or not processed_key:
            return 0.0
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([processed_student, processed_key])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Ensure value is between 0 and 1
        return max(0.0, min(1.0, float(similarity)))
        
    except Exception as e:
        logger.error(f"Error calculating relevance: {e}")
        return 0.0


def calculate_word_length_score(student_answer, expected_word_count):
    """
    Calculate word length score based on formula:
    word_length_score = min(student_word_count / expected_word_count, 1)
    Returns a value between 0 and 1.
    """
    try:
        # Count words in student answer
        student_words = student_answer.split()
        student_word_count = len(student_words)
        
        if expected_word_count <= 0:
            expected_word_count = 50  # Default minimum
        
        # Calculate score
        score = student_word_count / expected_word_count
        
        # Clamp to max of 1.0 (as per formula)
        return min(score, 1.0)
        
    except Exception as e:
        logger.error(f"Error calculating word length score: {e}")
        return 0.0


def calculate_grammar_score(text):
    """
    Calculate grammar score using simple heuristics.
    Formula: grammar_score = 1 - (grammar_errors / total_words)
    Returns a value between 0 and 1.
    
    Note: For better accuracy, consider integrating language_tool_python
    """
    try:
        if not text or not text.strip():
            return 0.0
        
        words = text.split()
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Simple grammar error detection (heuristic-based)
        grammar_errors = 0
        
        # Check for basic issues
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        for sentence in sentences:
            # Check if sentence starts with lowercase (likely error)
            if sentence and sentence[0].islower():
                grammar_errors += 1
            
            # Check for double spaces (formatting error)
            if '  ' in sentence:
                grammar_errors += 0.5
        
        # Check for missing punctuation at the end
        if text.strip() and text.strip()[-1] not in '.!?':
            grammar_errors += 1
        
        # Calculate score
        score = 1 - (grammar_errors / total_words)
        
        # Clamp between 0 and 1
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        logger.error(f"Error calculating grammar score: {e}")
        return 0.7  # Default moderate score


def calculate_spelling_score(text):
    """
    Calculate spelling score using simple heuristics.
    Formula: spelling_score = 1 - (spelling_errors / total_words)
    Returns a value between 0 and 1.
    
    Note: For better accuracy, consider integrating pyspellchecker or textblob
    """
    try:
        if not text or not text.strip():
            return 0.0
        
        words = [w for w in re.findall(r'\b[a-zA-Z]+\b', text)]
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Simple spelling error detection (heuristic-based)
        spelling_errors = 0
        
        for word in words:
            # Check for common issues
            # 1. Words with unusual character repetition (e.g., "helllo")
            if re.search(r'(.)\1{2,}', word):
                spelling_errors += 1
            
            # 2. Very short words (less than 2 chars) that aren't common
            if len(word) == 1 and word.lower() not in ['a', 'i']:
                spelling_errors += 0.5
            
            # 3. Words with unusual patterns (all caps in middle of sentence)
            if len(word) > 3 and word.isupper():
                spelling_errors += 0.2
        
        # Calculate score
        score = 1 - (spelling_errors / total_words)
        
        # Clamp between 0 and 1
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        logger.error(f"Error calculating spelling score: {e}")
        return 0.8  # Default good score


def generate_grading_justification(relevance, word_length, grammar, spelling, final_marks, max_marks):
    """Generate a brief justification for the grading"""
    justification_parts = []
    
    # Relevance feedback
    if relevance >= 0.8:
        justification_parts.append("Excellent relevance to answer key")
    elif relevance >= 0.6:
        justification_parts.append("Good relevance with minor gaps")
    elif relevance >= 0.4:
        justification_parts.append("Moderate relevance, some key points missing")
    else:
        justification_parts.append("Low relevance, many key concepts missing")
    
    # Word length feedback
    if word_length >= 1.0:
        justification_parts.append("adequate length")
    elif word_length >= 0.7:
        justification_parts.append("slightly short")
    else:
        justification_parts.append("significantly short answer")
    
    # Grammar feedback
    if grammar >= 0.9:
        justification_parts.append("excellent grammar")
    elif grammar >= 0.7:
        justification_parts.append("good grammar with minor errors")
    else:
        justification_parts.append("grammar needs improvement")
    
    # Spelling feedback
    if spelling >= 0.9:
        justification_parts.append("excellent spelling")
    elif spelling >= 0.7:
        justification_parts.append("good spelling with few errors")
    else:
        justification_parts.append("spelling needs improvement")
    
    percentage = (final_marks / max_marks * 100) if max_marks > 0 else 0
    
    return f"{'; '.join(justification_parts)}. Score: {final_marks}/{max_marks} ({percentage:.1f}%)"


def calculate_cosine_similarity(student_answer, answer_key):
    """
    Legacy function for backward compatibility.
    Now redirects to calculate_relevance().
    """
    return calculate_relevance(student_answer, answer_key)

def evaluate_answer(student_answer, answer_key, question_text, max_score, min_word_count):
    """
    Evaluate a student's answer using 4-parameter grading system:
    - Relevance (40%): Semantic similarity
    - Word Length (40%): Word count comparison
    - Grammar (10%): Grammar correctness
    - Spelling (10%): Spelling correctness
    """
    try:
        logger.info("Using 4-parameter grading system")
        
        # Calculate all four parameters
        relevance_score = calculate_relevance(student_answer, answer_key)
        word_length_score = calculate_word_length_score(student_answer, min_word_count)
        grammar_score = calculate_grammar_score(student_answer)
        spelling_score = calculate_spelling_score(student_answer)
        
        # Calculate final score using the formula
        final_score_0_to_1 = (
            (relevance_score * 0.40) +
            (word_length_score * 0.40) +
            (grammar_score * 0.10) +
            (spelling_score * 0.10)
        )
        
        # Convert to marks
        final_marks = round(final_score_0_to_1 * max_score, 2)
        
        # Generate justification
        feedback = generate_grading_justification(
            relevance_score, word_length_score, grammar_score, spelling_score, final_marks, max_score
        )
        
        logger.info(f"Grading complete: Rel={relevance_score:.2f}, WL={word_length_score:.2f}, "
                   f"Gram={grammar_score:.2f}, Spell={spelling_score:.2f}, Final={final_marks}/{max_score}")
        
        # Return: (relevance, word_length, feedback, final_marks)
        # Note: We return word_length_score in place of keyword_score for compatibility
        return relevance_score, word_length_score, feedback, final_marks
            
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        # Ultimate fallback with minimal scores
        return 0.1, 0.1, "Error in evaluation. Using minimal scores.", 0

def simple_evaluate_answer(student_answer, answer_key, keywords=None):
    """
    A simple fallback method for answer evaluation when NLP processing fails
    Uses basic string matching and word overlap
    """
    try:
        # First, clean any NULL bytes and control characters
        student_answer = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', student_answer)
        answer_key = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', answer_key)

        if keywords:
            keywords = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', keywords)

        # Simple word overlap score
        student_words = set(student_answer.lower().split())
        answer_words = set(answer_key.lower().split())

        # Remove very common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'is', 'are'}
        student_words = student_words - common_words
        answer_words = answer_words - common_words

        # Calculate simple overlap
        if not student_words or not answer_words:
            overlap_score = 0
        else:
            overlap = student_words.intersection(answer_words)
            overlap_score = len(overlap) / max(len(answer_words), 1)

        # Simple keyword check
        keyword_matched = []
        keyword_missing = []

        if keywords:
            keyword_list = [k.strip().lower() for k in keywords.split(',')]
            for kw in keyword_list:
                if kw in student_answer.lower():
                    keyword_matched.append(kw)
                else:
                    keyword_missing.append(kw)
        else:
            # Use important words from answer key as keywords
            keyword_list = [word for word in answer_words if len(word) > 4][:5]  # Use 5 longest words
            for kw in keyword_list:
                if kw in student_words:
                    keyword_matched.append(kw)
                else:
                    keyword_missing.append(kw)

        # Calculate keyword score
        if not keyword_list:
            keyword_score = overlap_score  # Fall back to overlap score
        else:
            keyword_score = len(keyword_matched) / len(keyword_list)

        # Generate simple feedback
        feedback = ""  # Remove feedback generation

        return overlap_score, keyword_score, feedback.strip()
    except Exception as e:
        # Ultimate fallback - if all else fails, return minimal scores
        logger.error(f"Fatal error in simple evaluation: {e}")
        return 0.1, 0.1, ""

def extract_student_info(text):
    """
    Attempt to extract student name and ID from the answer paper
    This is a simple implementation and may need customization for different formats
    """
    try:
        # Look for common patterns
        name_pattern = re.compile(r'name[:\s]+([A-Za-z\s]+)', re.IGNORECASE)
        id_pattern = re.compile(r'(id|roll|number)[:\s]+([A-Za-z0-9\-]+)', re.IGNORECASE)

        # Search for matches
        name_match = name_pattern.search(text)
        id_match = id_pattern.search(text)

        # Extract matches if found
        student_name = name_match.group(1).strip() if name_match else ""
        student_id = id_match.group(2).strip() if id_match else ""

        # Validate extracted info
        if student_name and len(student_name) > 50:  # Probably extracted too much text
            student_name = student_name[:50]

        if student_id and len(student_id) > 20:  # ID is usually shorter
            student_id = student_id[:20]

        return (student_name, student_id)
    except Exception as e:
        logger.error(f"Error extracting student info: {e}")
        return ("", "")