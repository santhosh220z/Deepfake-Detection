/**
 * DeepGuard AI — Frontend Logic
 * Handles file upload, preview, API communication, and results rendering.
 */

// ─── DOM Elements ──────────────────────────────────────────
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewArea = document.getElementById('preview-area');
const previewImage = document.getElementById('preview-image');
const previewVideo = document.getElementById('preview-video');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const fileTypeIcon = document.getElementById('file-type-icon');
const btnRemove = document.getElementById('btn-remove');
const btnAnalyze = document.getElementById('btn-analyze');
const btnText = document.querySelector('.btn-text');
const btnLoading = document.querySelector('.btn-loading');
const resultsSection = document.getElementById('results-section');
const btnNewAnalysis = document.getElementById('btn-new-analysis');

let selectedFile = null;

// ─── File Upload Handling ──────────────────────────────────

// Click to upload
uploadArea.addEventListener('click', () => fileInput.click());

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Handle selected file
function handleFile(file) {
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/') || /\.(mp4|avi|mov|mkv|webm|flv|wmv)$/i.test(file.name);

    if (!isImage && !isVideo) {
        showError('Please upload an image or video file.');
        return;
    }

    selectedFile = file;

    // Update file info
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileTypeIcon.textContent = isImage ? '🖼️' : '🎬';

    // Show preview
    previewImage.style.display = 'none';
    previewVideo.style.display = 'none';

    const url = URL.createObjectURL(file);

    if (isImage) {
        previewImage.src = url;
        previewImage.style.display = 'block';
    } else {
        previewVideo.src = url;
        previewVideo.style.display = 'block';
    }

    uploadArea.style.display = 'none';
    previewArea.style.display = 'block';
    btnAnalyze.disabled = false;

    // Hide results if visible
    resultsSection.style.display = 'none';
}

// Remove file
btnRemove.addEventListener('click', () => {
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewImage.style.display = 'none';
    previewVideo.src = '';
    previewVideo.style.display = 'none';
    previewArea.style.display = 'none';
    uploadArea.style.display = '';
    btnAnalyze.disabled = true;
    setAnalyzeLoading(false);
}

// ─── Analysis ──────────────────────────────────────────────

btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile) return;

    setAnalyzeLoading(true);
    resultsSection.style.display = 'none';

    try {
        const isVideo = selectedFile.type.startsWith('video/') || 
                        /\.(mp4|avi|mov|mkv|webm|flv|wmv)$/i.test(selectedFile.name);
        const endpoint = isVideo ? '/api/predict/video' : '/api/predict/image';

        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        showError(error.message || 'An error occurred during analysis.');
    } finally {
        setAnalyzeLoading(false);
    }
});

function setAnalyzeLoading(loading) {
    btnAnalyze.disabled = loading;
    btnText.style.display = loading ? 'none' : '';
    btnLoading.style.display = loading ? 'flex' : 'none';
}

// ─── Results Display ──────────────────────────────────────

function displayResults(result) {
    const isReal = result.prediction === 'Real';
    const isFake = result.prediction === 'Fake';

    // Verdict
    const verdictDiv = document.getElementById('result-verdict');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictText = document.getElementById('verdict-text');
    const verdictSubtitle = document.getElementById('verdict-subtitle');

    verdictDiv.className = 'result-verdict ' + (isReal ? 'real' : 'fake');
    verdictIcon.innerHTML = isReal ? '✓' : '⚠';
    verdictText.textContent = result.prediction.toUpperCase();
    verdictSubtitle.textContent = isReal
        ? 'This content appears to be authentic'
        : 'This content appears to be manipulated or AI-generated';

    // Confidence
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceFill = document.getElementById('confidence-fill');

    confidenceFill.style.width = '0%';
    confidenceFill.className = 'confidence-fill ' + (isReal ? 'real' : 'fake');

    // Animate confidence
    setTimeout(() => {
        confidenceFill.style.width = result.confidence + '%';
        animateCounter(confidenceValue, result.confidence, '%');
    }, 200);

    // Details
    document.getElementById('detail-filename').textContent = result.filename || '-';
    document.getElementById('detail-type').textContent = (result.type || '-').toUpperCase();
    document.getElementById('detail-raw-score').textContent = result.raw_probability?.toFixed(6) || '-';

    // Video-specific results
    const videoResults = document.getElementById('video-results');
    if (result.type === 'video' && result.frames_analyzed) {
        document.getElementById('stat-total-frames').textContent = result.frames_analyzed;
        document.getElementById('stat-real-frames').textContent = result.real_frames;
        document.getElementById('stat-fake-frames').textContent = result.fake_frames;

        const realPercent = (result.real_frames / result.frames_analyzed) * 100;
        document.getElementById('frame-bar-real').style.width = realPercent + '%';

        videoResults.style.display = 'block';
    } else {
        videoResults.style.display = 'none';
    }

    // Show results section with animation
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── New Analysis ──────────────────────────────────────────

btnNewAnalysis.addEventListener('click', () => {
    resetUpload();
    resultsSection.style.display = 'none';
    document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

// ─── Utilities ─────────────────────────────────────────────

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function animateCounter(element, target, suffix = '') {
    const duration = 1000;
    const startTime = performance.now();
    const startValue = 0;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
        const current = startValue + (target - startValue) * eased;
        element.textContent = current.toFixed(1) + suffix;

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = target.toFixed(1) + suffix;
        }
    }

    requestAnimationFrame(update);
}

function showError(message) {
    // Create a temporary error toast
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 24px;
        right: 24px;
        padding: 16px 24px;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        color: #f87171;
        font-family: var(--font);
        font-size: 0.88rem;
        font-weight: 500;
        z-index: 1000;
        backdrop-filter: blur(12px);
        animation: fadeSlideIn 0.3s ease;
        max-width: 400px;
    `;
    toast.textContent = '⚠ ' + message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
