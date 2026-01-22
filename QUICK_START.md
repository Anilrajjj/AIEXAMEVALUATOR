# AIExamEvaluator - Quick Start Guide

## 🚀 5-Minute Setup

### Prerequisites
1. **Python 3.11+** installed
2. **Tesseract OCR** installed ([Windows](https://github.com/UB-Mannheim/tesseract/wiki) | [Mac](https://formulae.brew.sh/formula/tesseract) | [Linux](https://tesseract-ocr.github.io/tessdoc/Installation.html))
3. **Poppler** installed (for PDF processing)

### Installation

```powershell
# 1. Clone the repository
git clone <your-repo-url>
cd AIExamEvaluator

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate    # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python main.py
```

### First Use

1. **Open browser**: http://localhost:5000
2. **Register**: Create a faculty account
3. **Create Exam**: Add exam title and questions
4. **Upload Answers**: Upload student answer papers (PDF/images)
5. **View Results**: Check scores and reports

---

## 📝 Basic Usage

### Creating an Exam

```
Dashboard → Create New Exam → Enter Details:
- Exam Title: "Midterm Exam 2025"
- Description: "Computer Networks"
- Click "Create Exam"
```

### Adding Questions

```
Edit Exam → Add Question:
- Question Text: "Explain TCP/IP model"
- Answer Key: "TCP/IP has 4 layers: Application, Transport..."
- Max Score: 10
- Min Word Count: 50
- Click "Add Question"
```

### Uploading Student Answers

```
Exam Dashboard → Upload Answers:
- Student Name: "John Doe"
- Student ID: "CS001"
- Upload File: Select PDF/image
- Click "Upload"
```

---

## 🔧 Configuration

### Environment Variables (Optional)

```powershell
# Windows PowerShell
$env:SESSION_SECRET = "your-secret-key-here"

# Linux/Mac
export SESSION_SECRET="your-secret-key-here"
```

### Adjust Grading Settings

Edit `utils/scoring.py`:
- Line 11: Change `PASSING_PERCENTAGE = 0.35` (35%)
- Modify scoring weights in `calculate_score()`

---

## 📊 Features at a Glance

| Feature | Description |
|---------|-------------|
| **OCR** | Extracts text from handwritten/typed papers |
| **TrOCR** | Deep learning model for handwriting |
| **Auto-Grading** | NLP-based answer evaluation |
| **Reports** | Detailed analytics and statistics |
| **Export** | JSON format reports |
| **Manual Entry** | Fallback if OCR fails |

---

## 🐛 Troubleshooting

### "Tesseract not found"
```powershell
# Windows: Add Tesseract to PATH
$env:PATH += ";C:\Program Files\Tesseract-OCR"

# Or set directly in code (app.py)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### "Cannot convert PDF"
```powershell
# Install Poppler and add to PATH
# Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/
# Extract and add bin folder to PATH
$env:PATH += ";C:\path\to\poppler\bin"
```

### Poor OCR Accuracy
- Use high-quality scans (300+ DPI)
- Ensure good lighting
- Avoid shadows and creases
- Try manual entry as fallback

### Memory Errors
- System automatically downscales large images
- PDFs limited to 10 pages (configurable)
- Close other applications for more RAM

---

## 📁 Project Structure (Simplified)

```
AIExamEvaluator/
├── app.py              # Flask app setup
├── main.py             # Run this to start
├── models.py           # Database models
├── routes/             # Web routes
├── utils/              # OCR, NLP, scoring
├── templates/          # HTML templates
├── static/             # CSS, JS, images
├── uploads/            # User uploads (gitignored)
└── instance/           # Database (gitignored)
```

---

## 🎯 Common Tasks

### Backup Database
```powershell
Copy-Item instance\app.db instance\app.db.backup
```

### Clear All Submissions
```sql
# Use SQLite browser or:
python
>>> from app import app, db
>>> with app.app_context():
>>>     db.session.execute('DELETE FROM submission_answer')
>>>     db.session.execute('DELETE FROM submission')
>>>     db.session.commit()
```

### View Logs
```powershell
# Enable logging in main.py (already enabled)
# Check console output
```

### Export Data
- Go to submission page
- Click "Download Report (JSON)"
- Save the file

---

## 📞 Getting Help

1. **Read Documentation**:
   - `README.md` - Full documentation
   - `PROJECT_ANALYSIS.md` - Technical deep dive
   - `CLEANUP_SUMMARY.md` - Project cleanup info

2. **Check Issues**:
   - Search existing GitHub issues
   - Create new issue with error logs

3. **Common Solutions**:
   - Restart the application
   - Clear browser cache
   - Check file permissions
   - Verify dependencies installed

---

## 🔐 Security Tips

- ❌ Never commit `.env` files
- ❌ Never commit database files
- ❌ Never commit student data
- ✅ Use strong SESSION_SECRET in production
- ✅ Change default credentials
- ✅ Use HTTPS in production

---

## 📈 Performance Tips

1. **For Fast OCR**:
   - Use printed text (PyTesseract is faster)
   - Enable TrOCR fast mode in code
   - Process smaller images

2. **For Accurate OCR**:
   - Use high DPI scans (600+)
   - Enable TrOCR accuracy mode
   - Preprocess images properly

3. **For Large Scale**:
   - Use PostgreSQL instead of SQLite
   - Enable database indexing
   - Use cloud storage for uploads
   - Add Redis caching

---

## 🎓 Learning Path

### For End Users
1. Read this Quick Start Guide
2. Watch demo video (if available)
3. Try with sample papers
4. Read best practices for scanning

### For Developers
1. Read README.md
2. Study PROJECT_ANALYSIS.md
3. Explore code in `routes/` and `utils/`
4. Set up development environment
5. Contribute improvements

---

## 🚀 What's Next?

### After Setup
1. ✅ Create your first exam
2. ✅ Upload a test answer paper
3. ✅ Review the evaluation
4. ✅ Generate a report
5. ✅ Explore analytics

### Future Features
- Integration with Google Classroom
- Mobile app
- Advanced AI grading (GPT-4)
- Multi-language support
- Plagiarism detection

---

## 📋 Checklist

### Before Using
- [ ] Python 3.11+ installed
- [ ] Tesseract OCR installed
- [ ] Poppler installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Application runs without errors

### First Time Setup
- [ ] Register faculty account
- [ ] Create test exam
- [ ] Add sample questions
- [ ] Upload test answer paper
- [ ] Verify OCR extraction
- [ ] Check grading results

### Production Deployment
- [ ] Set SESSION_SECRET
- [ ] Use PostgreSQL
- [ ] Enable HTTPS
- [ ] Set up backups
- [ ] Configure logging
- [ ] Monitor performance

---

## 💡 Pro Tips

1. **Better OCR Results**:
   - Scan at 300+ DPI
   - Use black ink
   - Avoid colored paper
   - Ensure even lighting

2. **Faster Grading**:
   - Pre-define answer keys clearly
   - Use keywords in answer keys
   - Set appropriate min word counts

3. **Better Analytics**:
   - Grade multiple submissions
   - Use consistent question types
   - Review common errors report

4. **Save Time**:
   - Reuse exams for future semesters
   - Export reports for record-keeping
   - Use manual entry for difficult cases

---

**Version**: 1.0  
**Last Updated**: 2025-01-28  
**Support**: Check GitHub issues or create new one

---

## 🎉 You're Ready!

Start the application and begin evaluating exams automatically!

```powershell
python main.py
# Open: http://localhost:5000
```

Happy Grading! 📝✨

