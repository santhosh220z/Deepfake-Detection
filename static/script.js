const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const chooseBtn = document.getElementById("choose-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const fileMeta = document.getElementById("file-meta");
const errorText = document.getElementById("error-text");
const dropTitle = document.getElementById("drop-title");
const dropSubtitle = document.getElementById("drop-subtitle");

const modeImageBtn = document.getElementById("mode-image");
const modeVideoBtn = document.getElementById("mode-video");

const previewImage = document.getElementById("preview-image");
const previewVideo = document.getElementById("preview-video");
const previewPlaceholder = document.getElementById("preview-placeholder");

const spinner = document.getElementById("spinner");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const resultMeta = document.getElementById("result-meta");
const predictionLabel = document.getElementById("prediction-label");
const predictionConfidence = document.getElementById("prediction-confidence");
const probabilityBars = document.getElementById("probability-bars");

const MODE_CONFIG = {
    image: {
        endpoint: "/predict",
        allowedExtensions: [".jpg", ".jpeg", ".png"],
        allowedMimeTypes: ["image/jpeg", "image/png"],
        accept: ".jpg,.jpeg,.png,image/jpeg,image/png",
        dropTitle: "Drop image here",
        dropSubtitle: "or click to browse JPG, JPEG, PNG files",
        previewPlaceholder: "Image preview appears here.",
        emptyResultText: "Upload and analyze an image to view label confidence and class probabilities.",
    },
    video: {
        endpoint: "/predict-video",
        allowedExtensions: [".mp4", ".avi", ".mov", ".mpeg", ".mpg", ".webm", ".m4v"],
        allowedMimeTypes: [
            "video/mp4",
            "video/x-msvideo",
            "video/quicktime",
            "video/mpeg",
            "video/webm",
            "application/octet-stream",
        ],
        accept: ".mp4,.avi,.mov,.mpeg,.mpg,.webm,.m4v,video/mp4,video/x-msvideo,video/quicktime,video/mpeg,video/webm",
        dropTitle: "Drop video here",
        dropSubtitle: "or click to browse MP4, AVI, MOV, MPEG, MPG, WEBM, M4V files",
        previewPlaceholder: "Video preview appears here.",
        emptyResultText: "Upload and analyze a video to view final label confidence and class probabilities.",
    },
};

let currentMode = "image";
let selectedFile = null;
let previewObjectUrl = null;

function formatFileSize(bytes) {
    if (bytes < 1024) {
        return `${bytes} B`;
    }
    const kb = bytes / 1024;
    if (kb < 1024) {
        return `${kb.toFixed(1)} KB`;
    }
    return `${(kb / 1024).toFixed(2)} MB`;
}

function setError(message) {
    errorText.textContent = message || "";
}

function clearError() {
    setError("");
}

function resetResult() {
    resultContent.classList.add("hidden");
    resultEmpty.classList.remove("hidden");
    resultEmpty.textContent = MODE_CONFIG[currentMode].emptyResultText;

    probabilityBars.innerHTML = "";
    predictionLabel.textContent = "-";
    predictionLabel.dataset.state = "unknown";
    predictionConfidence.textContent = "-";

    resultMeta.textContent = "";
    resultMeta.classList.add("hidden");
}

function resetPreview() {
    if (previewObjectUrl) {
        URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = null;
    }

    previewImage.removeAttribute("src");
    previewImage.style.display = "none";

    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.load();
    previewVideo.style.display = "none";

    previewPlaceholder.style.display = "block";
}

function setLoading(isLoading) {
    spinner.classList.toggle("hidden", !isLoading);
    chooseBtn.disabled = isLoading;
    analyzeBtn.disabled = isLoading || !selectedFile;
    dropZone.classList.toggle("is-disabled", isLoading);
    modeImageBtn.disabled = isLoading;
    modeVideoBtn.disabled = isLoading;
}

function prettifyLabel(label, mode) {
    const value = String(label || "").toLowerCase();
    if (value.includes("real")) {
        return {
            text: mode === "video" ? "Real Video" : "Real Image",
            state: "real",
        };
    }
    if (value.includes("fake")) {
        return {
            text: mode === "video" ? "Deepfake Video" : "Deepfake",
            state: "fake",
        };
    }
    if (!value) {
        return { text: "Unknown", state: "unknown" };
    }
    return {
        text: value.charAt(0).toUpperCase() + value.slice(1),
        state: "unknown",
    };
}

function createProbabilityRow(name, probability) {
    const row = document.createElement("div");
    row.className = "prob-row";

    const nameElement = document.createElement("span");
    nameElement.className = "prob-name";
    nameElement.textContent = name.charAt(0).toUpperCase() + name.slice(1);

    const track = document.createElement("div");
    track.className = "prob-track";

    const fill = document.createElement("div");
    fill.className = "prob-fill";
    fill.style.width = `${(probability * 100).toFixed(2)}%`;
    track.appendChild(fill);

    const valueElement = document.createElement("span");
    valueElement.className = "prob-value";
    valueElement.textContent = `${(probability * 100).toFixed(2)}%`;

    row.append(nameElement, track, valueElement);
    return row;
}

function detectModeFromFile(file) {
    if (!file) {
        return null;
    }

    const lowerName = file.name.toLowerCase();
    if (MODE_CONFIG.video.allowedExtensions.some((extension) => lowerName.endsWith(extension))) {
        return "video";
    }
    if (MODE_CONFIG.image.allowedExtensions.some((extension) => lowerName.endsWith(extension))) {
        return "image";
    }
    return null;
}

function validateFileForCurrentMode(file) {
    const cfg = MODE_CONFIG[currentMode];

    if (!file) {
        return currentMode === "video"
            ? "Please select a video file."
            : "Please select an image file.";
    }

    const lowerName = file.name.toLowerCase();
    const extensionOk = cfg.allowedExtensions.some((extension) => lowerName.endsWith(extension));
    const mimeTypeOk = file.type
        ? cfg.allowedMimeTypes.includes(file.type.toLowerCase())
        : true;

    if (!extensionOk || !mimeTypeOk) {
        return currentMode === "video"
            ? "Unsupported file type. Allowed: mp4, avi, mov, mpeg, mpg, webm, m4v."
            : "Unsupported file type. Allowed: jpg, jpeg, png.";
    }

    return "";
}

function applyPreview(file) {
    if (previewObjectUrl) {
        URL.revokeObjectURL(previewObjectUrl);
    }

    previewObjectUrl = URL.createObjectURL(file);

    if (currentMode === "video") {
        previewImage.style.display = "none";
        previewImage.removeAttribute("src");

        previewVideo.src = previewObjectUrl;
        previewVideo.style.display = "block";
        previewVideo.currentTime = 0;
    } else {
        previewVideo.pause();
        previewVideo.style.display = "none";
        previewVideo.removeAttribute("src");
        previewVideo.load();

        previewImage.src = previewObjectUrl;
        previewImage.style.display = "block";
    }

    previewPlaceholder.style.display = "none";
    fileMeta.textContent = `${file.name} • ${formatFileSize(file.size)}`;
}

function applyModeToUi() {
    const cfg = MODE_CONFIG[currentMode];

    modeImageBtn.classList.toggle("is-selected", currentMode === "image");
    modeVideoBtn.classList.toggle("is-selected", currentMode === "video");

    modeImageBtn.setAttribute("aria-selected", String(currentMode === "image"));
    modeVideoBtn.setAttribute("aria-selected", String(currentMode === "video"));

    fileInput.accept = cfg.accept;
    dropTitle.textContent = cfg.dropTitle;
    dropSubtitle.textContent = cfg.dropSubtitle;
    previewPlaceholder.textContent = cfg.previewPlaceholder;

    resetResult();
}

function switchMode(mode) {
    if (mode === currentMode) {
        return;
    }

    currentMode = mode;
    selectedFile = null;
    fileInput.value = "";
    fileMeta.textContent = mode === "video" ? "No video selected." : "No image selected.";
    analyzeBtn.disabled = true;

    clearError();
    resetPreview();
    applyModeToUi();
}

function setSelectedFile(file) {
    const detectedMode = detectModeFromFile(file);
    if (detectedMode && detectedMode !== currentMode) {
        switchMode(detectedMode);
    }

    const validationMessage = validateFileForCurrentMode(file);
    if (validationMessage) {
        setError(validationMessage);
        selectedFile = null;
        analyzeBtn.disabled = true;
        return;
    }

    selectedFile = file;
    clearError();
    applyPreview(file);
    analyzeBtn.disabled = false;
    resetResult();
}

function renderResult(payload) {
    const labelInfo = prettifyLabel(payload.label, currentMode);
    predictionLabel.textContent = labelInfo.text;
    predictionLabel.dataset.state = labelInfo.state;

    const confidence = Number(payload.confidence ?? 0);
    predictionConfidence.textContent = `${(confidence * 100).toFixed(2)}% confidence`;

    const classNames = Array.isArray(payload.class_names) && payload.class_names.length
        ? payload.class_names
        : ["real", "fake"];
    const probabilities = Array.isArray(payload.probabilities) ? payload.probabilities : [];

    const rows = classNames
        .map((name, index) => ({
            name: String(name || `class_${index}`),
            probability: Number(probabilities[index] ?? 0),
        }))
        .sort((a, b) => b.probability - a.probability);

    probabilityBars.innerHTML = "";
    rows.forEach((item) => {
        probabilityBars.appendChild(createProbabilityRow(item.name, item.probability));
    });

    if (currentMode === "video") {
        const metadata = [];
        if (payload.sampled_frames !== undefined) {
            metadata.push(`Sampled frames: ${payload.sampled_frames}`);
        }
        if (payload.face_detected_frames !== undefined) {
            metadata.push(`Face frames: ${payload.face_detected_frames}`);
        }
        resultMeta.textContent = metadata.join(" • ");
        resultMeta.classList.toggle("hidden", metadata.length === 0);
    } else {
        resultMeta.textContent = "";
        resultMeta.classList.add("hidden");
    }

    resultEmpty.classList.add("hidden");
    resultContent.classList.remove("hidden");
}

async function analyzeSelectedFile() {
    if (!selectedFile) {
        setError(
            currentMode === "video"
                ? "Please choose a video before running analysis."
                : "Please choose an image before running analysis."
        );
        return;
    }

    setLoading(true);
    clearError();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch(MODE_CONFIG[currentMode].endpoint, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        renderResult(data);
    } catch (error) {
        const message = error instanceof Error
            ? error.message
            : "Prediction failed. Please try again.";
        setError(message);
        resetResult();
    } finally {
        setLoading(false);
    }
}

modeImageBtn.addEventListener("click", () => switchMode("image"));
modeVideoBtn.addEventListener("click", () => switchMode("video"));

chooseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
    }
});

fileInput.addEventListener("change", () => {
    setSelectedFile(fileInput.files?.[0] || null);
});

["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("is-active");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("is-active");
    });
});

dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0] || null;
    setSelectedFile(file);
});

analyzeBtn.addEventListener("click", analyzeSelectedFile);

resetPreview();
applyModeToUi();
fileMeta.textContent = "No image selected.";
