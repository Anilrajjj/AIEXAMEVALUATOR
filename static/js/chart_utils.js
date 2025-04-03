/**
 * Chart utilities for creating and updating charts
 */

// Create a score distribution chart
function createScoreDistributionChart(chartId, labels, data) {
  const ctx = document.getElementById(chartId).getContext('2d');
  
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Number of Students',
        data: data,
        backgroundColor: 'rgba(75, 192, 192, 0.7)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            precision: 0
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Score Distribution'
        },
        legend: {
          display: false
        }
      }
    }
  });
}

// Create a question performance chart
function createQuestionPerformanceChart(chartId, questions, percentages) {
  const ctx = document.getElementById(chartId).getContext('2d');
  
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: questions,
      datasets: [{
        label: 'Average Score (%)',
        data: percentages,
        backgroundColor: 'rgba(54, 162, 235, 0.7)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Average Performance by Question'
        }
      }
    }
  });
}

// Create a radar chart for overall exam analysis
function createExamRadarChart(chartId, labels, data, maxScores) {
  const ctx = document.getElementById(chartId).getContext('2d');
  
  // Calculate percentages for normalized view
  const percentages = data.map((score, index) => {
    return (score / maxScores[index]) * 100;
  });
  
  return new Chart(ctx, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Average Score (%)',
        data: percentages,
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(255, 99, 132, 1)',
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: {
            display: true
          },
          suggestedMin: 0,
          suggestedMax: 100
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Question Performance Analysis'
        }
      }
    }
  });
}

// Load score distribution data via AJAX
function loadScoreDistributionData(examId, chartId) {
  fetch(`/api/exam/${examId}/score_distribution`)
    .then(response => response.json())
    .then(data => {
      createScoreDistributionChart(chartId, data.labels, data.counts);
    })
    .catch(error => {
      console.error('Error fetching score distribution data:', error);
    });
}

// Load question performance data via AJAX
function loadQuestionPerformanceData(examId, barChartId, radarChartId) {
  fetch(`/api/exam/${examId}/question_performance`)
    .then(response => response.json())
    .then(data => {
      createQuestionPerformanceChart(barChartId, data.questions, data.percentages);
      createExamRadarChart(radarChartId, data.questions, data.avg_scores, data.max_scores);
    })
    .catch(error => {
      console.error('Error fetching question performance data:', error);
    });
}
