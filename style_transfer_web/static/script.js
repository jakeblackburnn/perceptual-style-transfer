// Talks to the local server that served this page: relative URLs only.

const contentImageInput = document.getElementById('contentImage');
const contentPreview = document.getElementById('contentPreview');
const modelSelect = document.getElementById('modelSelect');
const transferBtn = document.getElementById('transferBtn');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('resultSection');
const resultPreview = document.getElementById('resultPreview');
const downloadBtn = document.getElementById('downloadBtn');
const errorDiv = document.getElementById('error');

// State: object URLs for the two previews, and the model that made the result.
let contentUrl = null;
let resultUrl = null;
let resultModelName = null;
let errorTimer = null;

// Replace a preview's image, revoking the object URL it showed before.
function showPreview(previewElement, oldUrl, blob, alt) {
    if (oldUrl) {
        URL.revokeObjectURL(oldUrl);
    }
    const url = URL.createObjectURL(blob);
    const img = document.createElement('img');
    img.src = url;
    img.alt = alt;
    previewElement.replaceChildren(img);
    return url;
}

function setSingleOption(text) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = text;
    modelSelect.replaceChildren(option);
}

// The server answers errors with JSON {"detail": "..."}.
async function errorMessage(response) {
    const data = await response.json().catch(() => ({}));
    return typeof data.detail === 'string' ? data.detail : `Request failed (${response.status})`;
}

async function loadModels() {
    try {
        const response = await fetch('/api/models');
        if (!response.ok) {
            throw new Error(await errorMessage(response));
        }
        const models = await response.json();

        if (models.length === 0) {
            setSingleOption('No models installed');
            showError('No models installed. Install or train a model, then reload this page. ' +
                'Run "style-transfer list" to see installed models.', false);
            return;
        }

        setSingleOption('Select a style...');
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = `${model.name} (${model.model_size})`;
            modelSelect.appendChild(option);
        });
    } catch (error) {
        setSingleOption('Error loading models');
        showError('Failed to load models: ' + error.message);
    }
}

function updateTransferButton() {
    const hasImage = contentImageInput.files.length > 0;
    const hasModel = modelSelect.value !== '';
    transferBtn.disabled = !(hasImage && hasModel);
}

contentImageInput.addEventListener('change', () => {
    const file = contentImageInput.files[0];
    if (file) {
        contentUrl = showPreview(contentPreview, contentUrl, file, 'Preview');
    }
    updateTransferButton();
});

modelSelect.addEventListener('change', updateTransferButton);

// One request: post the image and model name, get the stylized PNG back.
transferBtn.addEventListener('click', async () => {
    const contentFile = contentImageInput.files[0];
    const modelName = modelSelect.value;

    if (!contentFile) {
        showError('Please select a content image');
        return;
    }
    if (!modelName) {
        showError('Please select a style model');
        return;
    }

    transferBtn.disabled = true;
    loading.textContent = 'Applying style transfer...';
    loading.style.display = 'block';
    resultSection.style.display = 'none';
    errorDiv.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', contentFile);
        formData.append('model', modelName);

        const response = await fetch('/api/stylize', { method: 'POST', body: formData });
        if (!response.ok) {
            throw new Error(await errorMessage(response));
        }

        const blob = await response.blob();
        resultUrl = showPreview(resultPreview, resultUrl, blob, 'Styled Result');
        resultModelName = modelName;
        resultSection.style.display = 'block';
    } catch (error) {
        showError(`Error: ${error.message}`);
    } finally {
        loading.style.display = 'none';
        updateTransferButton();
    }
});

// Download the result that is on screen (same object URL as the preview).
downloadBtn.addEventListener('click', () => {
    if (!resultUrl) {
        showError('No image to download');
        return;
    }
    const a = document.createElement('a');
    a.href = resultUrl;
    a.download = `styled_${resultModelName}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
});

// Errors hide themselves after 5 seconds unless autoHide is false.
function showError(message, autoHide = true) {
    clearTimeout(errorTimer);
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    if (autoHide) {
        errorTimer = setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
}

loadModels();
