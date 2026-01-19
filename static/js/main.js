// Main JavaScript for Home Page

document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    const itemInput = document.getElementById('itemInput');
    const predictionResult = document.getElementById('predictionResult');

    if (predictionForm) {
        predictionForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const item = itemInput.value.trim();
            if (!item) {
                alert('Please enter an item to analyze');
                return;
            }

            // Show loading state
            predictionResult.classList.remove('hidden');
            predictionResult.innerHTML = '<div class="loading">Analyzing sustainability score...</div>';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ item: item })
                });

                const data = await response.json();

                if (data.success) {
                    displayPredictionResult(data);
                } else {
                    predictionResult.innerHTML = `<div class="error-message">Error: ${data.error || 'Failed to get prediction'}</div>`;
                }
            } catch (error) {
                predictionResult.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
            }
        });
    }
});

function displayPredictionResult(data) {
    const resultDiv = document.getElementById('predictionResult');
    
    // Validate and sanitize data
    if (!data || typeof data !== 'object') {
        resultDiv.innerHTML = '<div class="error-message">Invalid response data</div>';
        return;
    }
    
    // Determine score color
    const score = data.score || 0;
    let scoreColor = '#e74c3c'; // Low
    if (score >= 7) {
        scoreColor = '#2ecc71'; // High
    } else if (score >= 4) {
        scoreColor = '#f39c12'; // Medium
    }

    // Format confidence percentage
    const confidence = data.confidence || 0.5;
    const confidencePercent = (confidence * 100).toFixed(1);
    
    // Safely get score range
    let scoreRangeText = 'N/A';
    if (data.score_range) {
        if (Array.isArray(data.score_range) && data.score_range.length >= 2) {
            scoreRangeText = `${data.score_range[0]}-${data.score_range[1]}`;
        } else if (typeof data.score_range === 'object' && data.score_range.length !== undefined) {
            scoreRangeText = `${data.score_range[0]}-${data.score_range[1]}`;
        }
    }

    const html = `
        <div class="result-card" style="background: linear-gradient(135deg, ${scoreColor} 0%, ${adjustBrightness(scoreColor, -20)} 100%);">
            <div style="text-align: center;">
                <h3 style="margin-bottom: 0.5rem; opacity: 0.9;">${data.item}</h3>
                <div class="result-score" style="color: white;">${(data.score || 0).toFixed(1)}</div>
                <div class="result-category">${data.prediction || 'Medium'} Sustainability Score</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Score Range: ${scoreRangeText}</div>
            </div>
            
            <div class="result-details">
                <p><strong>Environmental Impact:</strong> ${data.environmental_impact || 'Medium'}</p>
                <p><strong>Recommendation:</strong> ${data.recommendation || 'Please dispose responsibly'}</p>
                <p style="margin-top: 1rem;"><strong>Explanation:</strong></p>
                <p style="font-size: 0.9rem; opacity: 0.95;">${data.explanation || 'Sustainability analysis completed'}</p>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>Confidence:</span>
                        <span>${confidencePercent}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidencePercent}%;"></div>
                    </div>
                </div>
                
                ${data.probabilities && typeof data.probabilities === 'object' && data.probabilities !== null && !Array.isArray(data.probabilities) ? `
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
                    <p style="font-size: 0.85rem; opacity: 0.9;"><strong>Probability Distribution:</strong></p>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem;">
                        ${(() => {
                            try {
                                const keys = Object.keys(data.probabilities || {});
                                return keys.map(key => {
                                    const value = data.probabilities[key] || 0;
                                    return `<span style="font-size: 0.85rem;">${key}: ${(value * 100).toFixed(1)}%</span>`;
                                }).join('');
                            } catch (e) {
                                return '';
                            }
                        })()}
                    </div>
                </div>
                ` : ''}
                
                ${data.using_rag_model ? `
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
                    <p style="font-size: 0.85rem; opacity: 0.9;"><strong>RAG Analysis:</strong></p>
                    <p style="font-size: 0.85rem; opacity: 0.9;">Analyzed using Retrieval-Augmented Generation model</p>
                    <p style="font-size: 0.85rem; opacity: 0.9;">RAG Confidence: ${((data.rag_confidence || 0) * 100).toFixed(1)}%</p>
                    ${data.rag_sources && typeof data.rag_sources === 'object' && data.rag_sources !== null ? 
                        `<p style="font-size: 0.85rem; opacity: 0.9;">Sources: ${data.rag_sources.product_matches || 0} product matches, ${data.rag_sources.knowledge_contexts || 0} knowledge contexts</p>` : ''}
                    ${data.rag_recommendations && Array.isArray(data.rag_recommendations) && data.rag_recommendations.length > 0 ? 
                        `<div style="margin-top: 0.5rem;">
                            <p style="font-size: 0.85rem; opacity: 0.9;"><strong>Recommendations:</strong></p>
                            ${(() => {
                                try {
                                    return data.rag_recommendations.slice(0, 3).map(rec => {
                                        if (!rec || typeof rec !== 'object') return '';
                                        const action = rec.action || '';
                                        const reason = rec.reason || '';
                                        return `<p style="font-size: 0.85rem; opacity: 0.9; margin-top: 0.25rem;">• ${action} ${reason}</p>`;
                                    }).filter(x => x).join('');
                                } catch (e) {
                                    return '';
                                }
                            })()}
                        </div>` : ''}
                </div>
                ` : ''}
            </div>
        </div>
    `;

    resultDiv.innerHTML = html;
}

function adjustBrightness(color, percent) {
    // Simple brightness adjustment for gradient
    const num = parseInt(color.replace("#", ""), 16);
    const amt = Math.round(2.55 * percent);
    const R = Math.min(255, Math.max(0, (num >> 16) + amt));
    const G = Math.min(255, Math.max(0, ((num >> 8) & 0x00FF) + amt));
    const B = Math.min(255, Math.max(0, (num & 0x0000FF) + amt));
    return "#" + (0x1000000 + R * 0x10000 + G * 0x100 + B).toString(16).slice(1);
}

