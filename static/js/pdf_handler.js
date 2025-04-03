/**
 * PDF handling utilities using PDF.js
 */

// Initialize PDF.js
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.4.120/build/pdf.worker.min.js';

// Render a PDF preview
function renderPdfPreview(url, canvasId, pageNum = 1) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Show loading
  ctx.font = '16px Arial';
  ctx.fillText('Loading PDF...', 50, 100);
  
  // Load the PDF document
  pdfjsLib.getDocument(url).promise.then(function(pdf) {
    // Get the requested page
    pdf.getPage(pageNum).then(function(page) {
      // Set scale
      const viewport = page.getViewport({ scale: 1.5 });
      canvas.height = viewport.height;
      canvas.width = viewport.width;
      
      // Render the PDF page
      const renderContext = {
        canvasContext: ctx,
        viewport: viewport
      };
      
      page.render(renderContext);
    });
  }).catch(function(error) {
    console.error('Error rendering PDF:', error);
    
    // Show error message
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '16px Arial';
    ctx.fillText('Error loading PDF', 50, 100);
  });
}

// Handle file input for PDF preview
function setupPdfFileInput(inputId, previewCanvasId) {
  const fileInput = document.getElementById(inputId);
  
  fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    
    if (file && file.type === 'application/pdf') {
      const fileUrl = URL.createObjectURL(file);
      renderPdfPreview(fileUrl, previewCanvasId);
    } else if (file) {
      // If it's not a PDF, check if it's an image
      if (file.type.startsWith('image/')) {
        const canvas = document.getElementById(previewCanvasId);
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Create an image object
        const img = new Image();
        img.onload = function() {
          // Set canvas dimensions to match image with max size
          const maxWidth = 800;
          const maxHeight = 600;
          let width = img.width;
          let height = img.height;
          
          if (width > maxWidth) {
            height = height * (maxWidth / width);
            width = maxWidth;
          }
          
          if (height > maxHeight) {
            width = width * (maxHeight / height);
            height = maxHeight;
          }
          
          canvas.width = width;
          canvas.height = height;
          
          // Draw the image
          ctx.drawImage(img, 0, 0, width, height);
        };
        
        img.src = URL.createObjectURL(file);
      } else {
        // Not a PDF or image
        alert('Please select a PDF or image file.');
      }
    }
  });
}

// Display a processed answer with highlighting
function displayProcessedAnswer(containerId, text, keywords) {
  const container = document.getElementById(containerId);
  
  // Basic sanitization
  const sanitizedText = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
  
  // If no keywords, just show the text
  if (!keywords || keywords.length === 0) {
    container.innerHTML = `<p>${sanitizedText}</p>`;
    return;
  }
  
  // Highlight keywords
  let highlightedText = sanitizedText;
  keywords.forEach(keyword => {
    // Create a case-insensitive regular expression
    const regex = new RegExp(`(${keyword})`, 'gi');
    highlightedText = highlightedText.replace(regex, '<span class="keyword-highlight">$1</span>');
  });
  
  container.innerHTML = `<p>${highlightedText}</p>`;
}

// Generate file download for reports
function generateReportDownload(data, filename) {
  const blob = new Blob([data], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  
  // Cleanup
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
}
