# AIExamEvaluator

An AI-powered exam evaluation system that uses OCR and Natural Language Processing to automatically grade handwritten and typed student answer papers.

## 🚀 Features

- **User Authentication**: Secure faculty login and registration system
- **Exam Management**: Create exams with multiple questions and answer keys
- **Smart OCR**: 
  - TrOCR deep learning model for handwritten text recognition
  - PyTesseract fallback for printed text
  - Google Lens-style text region detection
- **Intelligent Grading**:
  - NLP-based answer evaluation using TF-IDF and cosine similarity
  - Keyword matching and semantic analysis
  - Automatic score calculation based on answer quality
- **Comprehensive Reports**:
  - Detailed student performance reports
  - Statistical analytics (mean, median, standard deviation)
  - Score distribution charts
  - Question-wise performance analysis
- **Answer Segmentation**: Automatic detection and separation of multiple answers from a single document
- **Manual Entry**: Option to manually enter answers if OCR fails

## 📋 Technology Stack

### Backend
- **Framework**: Flask 3.1.0
- **Database**: SQLAlchemy with SQLite
- **Authentication**: Flask-Login

### OCR & ML
- **Deep Learning OCR**: TrOCR (microsoft/trocr-base-handwritten)
- **Traditional OCR**: PyTesseract
- **PDF Processing**: PyMuPDF (fitz), pdf2image, pdfminer.six
- **Image Processing**: OpenCV, Pillow

### NLP & Analysis
- **NLP**: NLTK (tokenization, stopwords, lemmatization)
- **ML**: scikit-learn (TF-IDF, cosine similarity)
- **Grammar Checking**: Heuristic-based (upgradeable to language-tool-python)
- **Spelling Checking**: Heuristic-based (upgradeable to pyspellchecker)

## 🛠️ Installation

### Prerequisites

1. **Python 3.11+** is required
2. **Tesseract OCR** must be installed on your system:
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

3. **Poppler** (for PDF processing):
   - **Windows**: Download from [GitHub](https://github.com/oschwartz10612/poppler-windows/releases/)
   - **macOS**: `brew install poppler`
   - **Linux**: `sudo apt-get install poppler-utils`

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd AIExamEvaluator
   ```

2. **Create a virtual environment**:
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or if using `pyproject.toml`:
   ```bash
   pip install -e .
   ```

4. **Set environment variables** (optional):
   ```powershell
   # Windows PowerShell
   $env:SESSION_SECRET = "your-secret-key-here"
   
   # Linux/macOS
   export SESSION_SECRET="your-secret-key-here"
   ```

5. **Initialize the database**:
   The database will be automatically created when you first run the application.

6. **Run the application**:
   ```bash
   python main.py
   ```

7. **Access the application**:
   Open your browser and navigate to: `http://localhost:5000`

## 📁 Project Structure

```
AIExamEvaluator/
├── app.py                  # Flask app initialization
├── main.py                 # Application entry point
├── models.py               # Database models
├── forms.py                # WTForms form definitions
├── auth.py                 # Authentication routes
├── routes/
│   ├── exam_routes.py      # Exam management routes
│   └── report_routes.py    # Report generation routes
├── utils/
│   ├── ocr_processor.py    # OCR processing logic
│   ├── trocr_processor.py  # TrOCR deep learning pipeline
│   ├── gemini_processor.py # Gemini API (disabled)
│   ├── nlp_processor.py    # NLP evaluation
│   ├── gemini_grading.py   # Local heuristic grading
│   ├── aligner.py          # Text segment alignment
│   └── scoring.py          # Scoring algorithms
├── templates/              # HTML templates
├── static/                 # CSS, JS, and images
├── uploads/                # Uploaded files (gitignored)
├── instance/               # Database files (gitignored)
└── requirements.txt        # Python dependencies
```

## 💡 Usage Guide

### 1. Create an Exam

1. Register or log in as faculty
2. Click **"Create New Exam"**
3. Enter exam title and description
4. Add questions with:
   - Question text
   - Answer key (model answer)
   - Maximum score
   - Minimum word count
   - Question type (text, MCQ, short)

### 2. Upload Student Answers

1. Go to the exam dashboard
2. Click **"Upload Answer Papers"**
3. Upload PDF or image files containing student answers
4. The system will:
   - Extract text using OCR
   - Segment into individual answers
   - Evaluate each answer
   - Calculate total score

### 3. Review Submissions

1. View all submissions from the exam dashboard
2. Click on individual submissions to see:
   - Detailed scores per question
   - Similarity scores and feedback
   - Overall grade and statistics
3. Export reports as JSON

### 4. Analytics

- View score distribution charts
- Analyze question-wise performance
- Identify common errors and missing keywords
- Track class statistics (mean, median, std dev)

## 🔧 Configuration

### OCR Settings

The system uses two OCR engines with automatic fallback:

1. **TrOCR** (Primary for handwritten text)
   - High accuracy deep learning model
   - Slower but more accurate
   - Automatically enabled if PyTorch and Transformers are installed

2. **PyTesseract** (Fallback)
   - Fast traditional OCR
   - Works for printed and simple handwritten text
   - Used when TrOCR is unavailable or fails

### Grading Settings

Edit `utils/scoring.py` to adjust:
- **Passing percentage**: Default 35% (line 11)
- **Score calculation weights**
- **Minimum word counts**

### Grammar & Spelling Upgrade (Optional)

The system uses **heuristic-based** grammar and spelling checking by default. For better accuracy in production:

**Install advanced libraries**:
```bash
pip install language-tool-python  # 90-95% grammar accuracy
pip install pyspellchecker         # 95-98% spelling accuracy
```

**Update functions in `utils/nlp_processor.py`** as documented in `GRADING_SYSTEM.md`.

**Improvement**: Accuracy increases from ~30% to **90%+** for both grammar and spelling.

## 🚨 Troubleshooting

### Common Issues

1. **"Tesseract not found"**
   - Install Tesseract OCR
   - Add Tesseract to your system PATH

2. **"Cannot convert PDF to images"**
   - Install Poppler
   - Add Poppler bin directory to PATH

3. **Low OCR accuracy**
   - Ensure uploaded images are high quality (300+ DPI)
   - Use clear, well-lit scans
   - Avoid shadows and distortions

4. **Memory errors with large PDFs**
   - The system processes PDFs page by page
   - Large files are automatically downscaled
   - Consider splitting very large PDFs

## 📊 Grading Algorithm

The system uses a comprehensive **4-Parameter Grading System**:

1. **Relevance (40%)**: Semantic similarity using TF-IDF cosine similarity (scikit-learn)
2. **Word Length (40%)**: Compares student word count to expected length
3. **Grammar (10%)**: Grammatical correctness using heuristic rules (upgradeable to LanguageTool)
4. **Spelling (10%)**: Spelling accuracy using pattern matching (upgradeable to PySpellChecker)

### Formula
```
final_score = (relevance × 0.40) + (word_length × 0.40) + (grammar × 0.10) + (spelling × 0.10)
final_marks = final_score × max_marks
```

### Implementation Details
- **Relevance**: TF-IDF vectorization + cosine similarity (NLTK + scikit-learn)
- **Word Length**: Simple word count comparison with min(student/expected, 1.0)
- **Grammar**: Currently heuristic-based (~30% accuracy) - detects capitalization, punctuation
- **Spelling**: Currently heuristic-based (~30% accuracy) - detects repetition, patterns

All grading is performed **locally without external API calls**. For production use, upgrade grammar/spelling checkers to achieve **90%+ accuracy**. See `GRADING_SYSTEM.md` for detailed documentation and upgrade instructions.

## 🔐 Security Notes

- Never commit API keys or secrets to version control
- The `.gitignore` file excludes sensitive files
- Change the default `SESSION_SECRET` in production
- Database is stored in `instance/` directory (gitignored)

## 📝 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

**Note**: This system uses local NLP-based grading. For more sophisticated evaluation in production, consider integrating with advanced AI APIs (OpenAI GPT-4, Anthropic Claude, Google Gemini).
