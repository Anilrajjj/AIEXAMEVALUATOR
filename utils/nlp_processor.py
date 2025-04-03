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
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text for NLP analysis:
    - Convert to lowercase
    - Remove punctuation
    - Remove stopwords
    - Lemmatize words
    """
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
    """
    Evaluate a student's answer against the answer key
    Returns a tuple of (similarity_score, keyword_score, feedback)
    """
    # For very short answers, keyword matching may be sufficient
    keyword_result = keyword_matching(student_answer, answer_key, keywords)
    matching_keywords, missing_keywords, keyword_score = keyword_result
    
    # Calculate similarity for text answers
    if question_type == 'text':
        similarity_score = calculate_cosine_similarity(student_answer, answer_key)
    else:
        similarity_score = keyword_score  # For non-text questions, use keyword score
    
    # Generate feedback
    feedback = ""
    if matching_keywords:
        feedback += f"Your answer includes key concepts: {', '.join(matching_keywords)}. "
    
    if missing_keywords:
        feedback += f"Consider including: {', '.join(missing_keywords)}."
    
    return (similarity_score, keyword_score, feedback.strip())

def extract_student_info(text):
    """
    Attempt to extract student name and ID from the answer paper
    This is a simple implementation and may need customization for different formats
    """
    # Look for common patterns
    name_pattern = re.compile(r'name[:\s]+([A-Za-z\s]+)', re.IGNORECASE)
    id_pattern = re.compile(r'(id|roll|number)[:\s]+([A-Za-z0-9\-]+)', re.IGNORECASE)
    
    # Search for matches
    name_match = name_pattern.search(text)
    id_match = id_pattern.search(text)
    
    # Extract matches if found
    student_name = name_match.group(1).strip() if name_match else ""
    student_id = id_match.group(2).strip() if id_match else ""
    
    return (student_name, student_id)
