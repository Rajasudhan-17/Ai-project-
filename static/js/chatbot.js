// Chatbot JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');
    const suggestionChips = document.querySelectorAll('.chip');

    // Handle form submission
    if (chatForm) {
        chatForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const message = chatInput.value.trim();
            if (message) {
                addUserMessage(message);
                chatInput.value = '';
                await sendMessage(message);
            }
        });
    }

    // Handle suggestion chips
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', function() {
            const question = this.getAttribute('data-question');
            chatInput.value = question;
            chatForm.dispatchEvent(new Event('submit'));
        });
    });

    // Auto-focus input
    if (chatInput) {
        chatInput.focus();
    }
});

function addUserMessage(message) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${escapeHtml(message)}</p>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addBotMessage(message, sources = [], usingWatson = false) {
    const chatMessages = document.getElementById('chatMessages');
    
    // Remove typing indicator if present
    const typingIndicator = chatMessages.querySelector('.typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(0,0,0,0.1);">
                <p style="font-size: 0.85rem; color: #666;"><strong>Sources:</strong></p>
                ${sources.map(source => {
                    if (typeof source === 'string') {
                        return `<p style="font-size: 0.85rem; color: #666; margin-top: 0.25rem;">• ${source}</p>`;
                    } else {
                        return `<p style="font-size: 0.85rem; color: #666; margin-top: 0.25rem;">
                            • ${source.topic || source} ${source.sdg_target ? `(${source.sdg_target})` : ''}
                        </p>`;
                    }
                }).join('')}
            </div>
        `;
    }
    
    let watsonBadge = '';
    if (usingWatson) {
        watsonBadge = '<span style="font-size: 0.7rem; color: #0066cc; margin-left: 0.5rem;">🤖 Powered by IBM Granite (Open-Source)</span>';
    }

    messageDiv.innerHTML = `
        <div class="message-content">
            ${formatMessage(message)}
            ${sourcesHtml}
            <div class="message-timestamp">
                ${new Date().toLocaleTimeString()}${watsonBadge}
            </div>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function showTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-content">
            <div style="display: flex; gap: 0.5rem;">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

async function sendMessage(message) {
    showTypingIndicator();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        if (data.success) {
            addBotMessage(data.response, data.sources || [], data.using_watson || false);
        } else {
            addBotMessage(`Sorry, I encountered an error: ${data.error || 'Unknown error'}`);
        }
    } catch (error) {
        addBotMessage(`Sorry, I couldn't process your request. Please try again.`);
        console.error('Error:', error);
    }
}

function formatMessage(message) {
    // Convert newlines to <br>
    let formatted = escapeHtml(message).replace(/\n/g, '<br>');
    
    // Format lists (simple detection)
    formatted = formatted.replace(/^• (.+)$/gm, '<li>$1</li>');
    if (formatted.includes('<li>')) {
        formatted = '<ul>' + formatted + '</ul>';
    }
    
    return formatted;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

