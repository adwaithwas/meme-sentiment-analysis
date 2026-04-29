/**
 * Meme Sentiment Analyzer — Frontend Logic
 * Handles drag-and-drop, file upload, AJAX requests, and result rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const removeBtn = document.getElementById('removeBtn');
    const textInput = document.getElementById('textInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    const errorToast = document.getElementById('errorToast');
    const errorMessage = document.getElementById('errorMessage');

    let selectedFile = null;

    // ============================================================
    // DRAG & DROP
    // ============================================================
    
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    // ============================================================
    // FILE HANDLING
    // ============================================================

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file (JPG, PNG, GIF, WebP)');
            return;
        }
        if (file.size > 16 * 1024 * 1024) {
            showError('File too large! Maximum size is 16MB.');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            dropZone.style.display = 'none';
        };
        reader.readAsDataURL(file);

        updateButtonState();
    }

    removeBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        dropZone.style.display = 'block';
        updateButtonState();
    });

    // Enable button when file or text is available
    textInput.addEventListener('input', updateButtonState);

    function updateButtonState() {
        analyzeBtn.disabled = !(selectedFile || textInput.value.trim());
    }

    // ============================================================
    // ANALYZE
    // ============================================================

    analyzeBtn.addEventListener('click', async () => {
        if (analyzeBtn.disabled) return;

        // Toggle loading state
        const btnText = analyzeBtn.querySelector('.btn-text');
        const btnLoader = analyzeBtn.querySelector('.btn-loader');
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-flex';
        analyzeBtn.disabled = true;

        try {
            const formData = new FormData();
            if (selectedFile) formData.append('file', selectedFile);
            if (textInput.value.trim()) formData.append('text', textInput.value.trim());

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.error) {
                showError(data.error);
            } else if (data.success) {
                displayResults(data.results);
            }
        } catch (err) {
            showError('Connection error. Is the server running?');
            console.error(err);
        } finally {
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
            updateButtonState();
        }
    });

    // ============================================================
    // DISPLAY RESULTS
    // ============================================================

    function displayResults(results) {
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Sentiment
        renderCard('sentiment', results.sentiment, {
            positive: 'var(--accent-green)',
            neutral: 'var(--text-muted)',
            negative: 'var(--accent-red)',
        });

        // Humor
        renderCard('humor', results.humor, {
            not_funny: 'var(--text-muted)',
            funny: 'var(--accent-orange)',
            very_funny: 'var(--accent-primary)',
            hilarious: 'var(--accent-pink)',
        });

        // Sarcasm
        renderCard('sarcasm', results.sarcasm, {
            not_sarcastic: 'var(--accent-green)',
            sarcastic: 'var(--accent-pink)',
        });
    }

    function renderCard(task, result, colorMap) {
        // Set prediction badge
        const badge = document.getElementById(`${task}Badge`);
        badge.textContent = `${result.prediction} (${result.confidence})`;
        badge.className = `prediction-badge ${result.prediction}`;

        // Render confidence bars
        const barsContainer = document.getElementById(`${task}Bars`);
        barsContainer.innerHTML = '';

        for (const [label, prob] of Object.entries(result.probabilities)) {
            const pctValue = parseFloat(prob);
            const color = colorMap[label] || 'var(--accent-primary)';

            const barItem = document.createElement('div');
            barItem.className = 'bar-item';
            barItem.innerHTML = `
                <span class="bar-label">${label.replace(/_/g, ' ')}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width: 0%; background: ${color};"></div>
                </div>
                <span class="bar-value">${prob}</span>
            `;
            barsContainer.appendChild(barItem);

            // Animate bar fill
            requestAnimationFrame(() => {
                setTimeout(() => {
                    barItem.querySelector('.bar-fill').style.width = `${pctValue}%`;
                }, 100);
            });
        }
    }

    // ============================================================
    // ERROR HANDLING
    // ============================================================

    function showError(message) {
        errorMessage.textContent = message;
        errorToast.style.display = 'flex';
        setTimeout(() => {
            errorToast.style.display = 'none';
        }, 5000);
    }
});
