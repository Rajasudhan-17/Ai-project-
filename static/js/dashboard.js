// Dashboard JavaScript

let categoryChart = null;
let impactChart = null;

document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();
    loadHistory();
    
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', clearHistory);
    }
});

async function loadDashboardData() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        if (data.success) {
            updateStats(data.stats);
            createCharts(data.stats);
        }
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

function updateStats(stats) {
    // Update stat cards
    const totalItemsEl = document.getElementById('totalItems');
    const recyclableCountEl = document.getElementById('recyclableCount');
    const avgScoreEl = document.getElementById('avgScore');
    const predictionCountEl = document.getElementById('predictionCount');

    if (totalItemsEl) totalItemsEl.textContent = stats.total_items || 0;
    if (recyclableCountEl) recyclableCountEl.textContent = stats.recyclable_count || 0;
    if (avgScoreEl) avgScoreEl.textContent = (stats.avg_sustainability_score || 0).toFixed(1);
    if (predictionCountEl) predictionCountEl.textContent = stats.prediction_count || 0;
}

function createCharts(stats) {
    // Category Distribution Chart
    const categoryCtx = document.getElementById('categoryChart');
    if (categoryCtx && stats.categories) {
        const categoryData = Object.entries(stats.categories);
        
        if (categoryChart) {
            categoryChart.destroy();
        }

        categoryChart = new Chart(categoryCtx, {
            type: 'doughnut',
            data: {
                labels: categoryData.map(([label]) => label),
                datasets: [{
                    data: categoryData.map(([, value]) => value),
                    backgroundColor: [
                        '#2ecc71',
                        '#3498db',
                        '#f39c12',
                        '#e74c3c',
                        '#9b59b6',
                        '#1abc9c'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    // Environmental Impact Distribution Chart
    const impactCtx = document.getElementById('impactChart');
    if (impactCtx && stats.environmental_impact_dist) {
        const impactData = Object.entries(stats.environmental_impact_dist);
        
        if (impactChart) {
            impactChart.destroy();
        }

        impactChart = new Chart(impactCtx, {
            type: 'bar',
            data: {
                labels: impactData.map(([label]) => label),
                datasets: [{
                    label: 'Count',
                    data: impactData.map(([, value]) => value),
                    backgroundColor: [
                        '#2ecc71',
                        '#f39c12',
                        '#e74c3c',
                        '#c0392b'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.success) {
            displayHistory(data.history);
        }
    } catch (error) {
        console.error('Error loading history:', error);
        document.getElementById('historyList').innerHTML = '<p class="loading">Error loading history</p>';
    }
}

function displayHistory(history) {
    const historyList = document.getElementById('historyList');
    
    if (!history || history.length === 0) {
        historyList.innerHTML = `
            <div class="empty-history">
                <div class="empty-history-icon">📭</div>
                <p>No predictions yet. Start analyzing items on the home page!</p>
            </div>
        `;
        return;
    }

    // Sort by timestamp (newest first)
    const sortedHistory = [...history].sort((a, b) => 
        new Date(b.timestamp) - new Date(a.timestamp)
    );

    historyList.innerHTML = sortedHistory.map(item => {
        const date = new Date(item.timestamp);
        const badgeClass = item.prediction === 'High' ? 'badge-high' : 
                          item.prediction === 'Medium' ? 'badge-medium' : 'badge-low';
        
        return `
            <div class="history-item">
                <div class="history-item-header">
                    <span class="history-item-name">${escapeHtml(item.item)}</span>
                    <span class="history-item-time">${date.toLocaleString()}</span>
                </div>
                <div class="history-item-details">
                    <span class="history-badge ${badgeClass}">${item.prediction} (${item.score_range[0]}-${item.score_range[1]})</span>
                    <span style="font-size: 0.85rem; color: #666;">Confidence: ${(item.confidence * 100).toFixed(1)}%</span>
                </div>
            </div>
        `;
    }).join('');
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear all prediction history?')) {
        return;
    }

    try {
        const response = await fetch('/api/clear_history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();

        if (data.success) {
            loadHistory();
        } else {
            alert('Error clearing history: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        alert('Error clearing history');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

