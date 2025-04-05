/**
 * PDF handling utilities using PDF.js
 */

// Initialize PDF.js
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.4.120/build/pdf.worker.min.js';

// Render a PDF preview with multi-page support
function renderPdfPreview(url, containerId, initialPage = 1) {
  const container = document.getElementById(containerId);
  
  // Clear container
  container.innerHTML = '<div class="text-center p-4"><div class="spinner-border" role="status"></div><p class="mt-2">Loading PDF...</p></div>';
  
  // Create a container for pages and navigation
  const pdfContainer = document.createElement('div');
  pdfContainer.className = 'pdf-container';
  
  // Load the PDF document
  pdfjsLib.getDocument(url).promise.then(function(pdf) {
    // Store total pages
    const totalPages = pdf.numPages;
    
    // Create navigation controls if multiple pages
    if (totalPages > 1) {
      const navDiv = document.createElement('div');
      navDiv.className = 'pdf-navigation d-flex justify-content-between align-items-center mb-3';
      
      // Previous button
      const prevBtn = document.createElement('button');
      prevBtn.className = 'btn btn-sm btn-outline-primary';
      prevBtn.innerHTML = '<i class="fas fa-arrow-left"></i> Previous';
      prevBtn.disabled = initialPage <= 1;
      
      // Page indicator
      const pageIndicator = document.createElement('span');
      pageIndicator.className = 'page-indicator';
      pageIndicator.textContent = `Page ${initialPage} of ${totalPages}`;
      
      // Next button
      const nextBtn = document.createElement('button');
      nextBtn.className = 'btn btn-sm btn-outline-primary';
      nextBtn.innerHTML = 'Next <i class="fas fa-arrow-right"></i>';
      nextBtn.disabled = initialPage >= totalPages;
      
      // Add everything to navigation div
      navDiv.appendChild(prevBtn);
      navDiv.appendChild(pageIndicator);
      navDiv.appendChild(nextBtn);
      
      // Clear container and add navigation
      container.innerHTML = '';
      container.appendChild(navDiv);
      
      // Create container for the canvas
      const canvasContainer = document.createElement('div');
      canvasContainer.className = 'canvas-container border rounded';
      container.appendChild(canvasContainer);
      
      // Create canvas element
      const canvas = document.createElement('canvas');
      canvas.id = `${containerId}-canvas`;
      canvasContainer.appendChild(canvas);
      
      // Current page tracker
      let currentPage = initialPage;
      
      // Function to render a specific page
      const renderPage = (pageNum) => {
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Show loading
        ctx.font = '16px Arial';
        ctx.fillText('Loading page...', 50, 100);
        
        // Update navigation buttons and indicator
        prevBtn.disabled = pageNum <= 1;
        nextBtn.disabled = pageNum >= totalPages;
        pageIndicator.textContent = `Page ${pageNum} of ${totalPages}`;
        
        // Get and render the page
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
      };
      
      // Set up event listeners for navigation
      prevBtn.addEventListener('click', function() {
        if (currentPage > 1) {
          currentPage--;
          renderPage(currentPage);
        }
      });
      
      nextBtn.addEventListener('click', function() {
        if (currentPage < totalPages) {
          currentPage++;
          renderPage(currentPage);
        }
      });
      
      // Render initial page
      renderPage(currentPage);
    } else {
      // For single-page PDFs, use the simpler display
      container.innerHTML = '';
      
      // Create canvas element
      const canvas = document.createElement('canvas');
      canvas.id = `${containerId}-canvas`;
      container.appendChild(canvas);
      
      const ctx = canvas.getContext('2d');
      
      // Show loading
      ctx.font = '16px Arial';
      ctx.fillText('Loading PDF...', 50, 100);
      
      // Get and render the page
      pdf.getPage(1).then(function(page) {
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
    }
  }).catch(function(error) {
    console.error('Error rendering PDF:', error);
    
    // Show error message
    container.innerHTML = '<div class="alert alert-danger" role="alert"><i class="fas fa-exclamation-triangle me-2"></i>Error loading PDF: ' + error.message + '</div>';
  });
}

// Handle file input for PDF preview
function setupPdfFileInput(inputId, previewContainerId) {
  const fileInput = document.getElementById(inputId);
  
  fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    
    if (file && file.type === 'application/pdf') {
      const fileUrl = URL.createObjectURL(file);
      renderPdfPreview(fileUrl, previewContainerId);
    } else if (file) {
      // If it's not a PDF, check if it's an image
      if (file.type.startsWith('image/')) {
        const container = document.getElementById(previewContainerId);
        container.innerHTML = '';
        
        // Create canvas for the image
        const canvas = document.createElement('canvas');
        container.appendChild(canvas);
        const ctx = canvas.getContext('2d');
        
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
