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
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
        logger.info(f"NLTK resource '{resource}' is already available")
    except LookupError:
        logger.info(f"Downloading NLTK resource '{resource}'")
        nltk.download(resource)

# Make sure punkt_tab is available
try:
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    logger.info("Downloading punkt_tab")
    nltk.download('punkt')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text for NLP analysis:
    - Remove NUL bytes and control characters
    - Convert to lowercase
    - Remove punctuation
    - Remove stopwords
    - Lemmatize words
    """
    # First, clean any NULL bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

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
        answer_tokens = word_tokenize(answer_key.lower())
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

def calculate_cosine_similarity(student_answer, answer_key):
    """Calculate cosine similarity between student answer and answer key"""
    # Preprocess texts
    processed_student = preprocess_text(student_answer)
    processed_key = preprocess_text(answer_key)

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_student, processed_key])

        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except ValueError as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0

def evaluate_answer(student_answer, answer_key, keywords=None, question_type='text'):
    try:
        # First, clean any NULL bytes and control characters before processing
        student_answer = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', student_answer)
        answer_key = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', answer_key)

        if keywords:
            keywords = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', keywords)

        # For very short answers, keyword matching may be sufficient
        keyword_result = keyword_matching(student_answer, answer_key, keywords)
        matching_keywords, missing_keywords, keyword_score = keyword_result

        # Calculate similarity for text answers with additional semantic comparison
        if question_type == 'text':
            similarity_score = calculate_cosine_similarity(student_answer, answer_key)
            # Add answer length information to feedback
            word_count = len(student_answer.split())

        # Generate feedback
        feedback = ""
        if matching_keywords:
            feedback += f"Your answer includes key concepts: {', '.join(matching_keywords)}. "

        if missing_keywords:
            feedback += f"Consider including: {', '.join(missing_keywords)}."

        return (similarity_score, keyword_score, feedback.strip())
    except Exception as e:
        # Log error and use fallback method
        logger.error(f"Error in NLP evaluation: {e}. Using fallback method.")
        return simple_evaluate_answer(student_answer, answer_key, keywords)

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
        feedback = ""
        if keyword_matched:
            feedback += f"Key concepts found: {', '.join(keyword_matched)}. "

        if keyword_missing:
            feedback += f"Consider including: {', '.join(keyword_missing)}."

        return overlap_score, keyword_score, feedback.strip()
    except Exception as e:
        # Ultimate fallback - if all else fails, return minimal scores
        logger.error(f"Fatal error in simple evaluation: {e}")
        return 0.1, 0.1, "Error evaluating answer. Please review manually."

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