const fileInput = document.getElementById('file-input');
const dragDropArea = document.getElementById('drag-drop-area');
const previewArea = document.getElementById('preview-area');
const uploadStatus = document.getElementById('upload-status');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const progressBar = document.getElementById('progress-bar');
const progressText = document.querySelector('.progress-text');
const checkIcon = document.getElementById('check-icon');
const checkButton = document.getElementById('check-button');
const or = document.getElementById('or');
const cancelUploadButton = document.getElementById('cancel-upload');
const message = document.getElementById('message');
const uploadForm = document.getElementById('upload-form');
const imageDataInput = document.getElementById('image-data');
const testt = document.getElementById('testt');

let uploadInterval;

dragDropArea.addEventListener('click', () => {
    fileInput.click();
});

dragDropArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    dragDropArea.classList.add('dragover');
});

dragDropArea.addEventListener('dragleave', () => {
    dragDropArea.classList.remove('dragover');
});

dragDropArea.addEventListener('drop', (event) => {
    event.preventDefault();
    dragDropArea.classList.remove('dragover');
    const files = event.dataTransfer.files;
    handleFiles(files);
});

fileInput.addEventListener('change', (event) => {
    const files = event.target.files;
    handleFiles(files);
});

cancelUploadButton.addEventListener('click', cancelUpload);

function cancelUpload() {
    clearInterval(uploadInterval);
    uploadStatus.style.display = 'none';
    dragDropArea.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = '0%';
    previewArea.innerHTML = '';
    fileInput.value = '';
    checkButton.style.display = 'none';
    or.style.display = 'block';
    checkIcon.style.display = 'none';
    message.style.display = 'none';
    testt.style.display = 'none';
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            uploadFile(file);
        } else {
            alert('Please upload only image files.');
        }
    }
}

function uploadFile(file) {
    fileName.textContent = file.name;
    fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
    uploadStatus.style.display = 'flex';
    progressBar.style.width = '0%';
    progressText.textContent = '0%';
    checkIcon.style.display = 'none';
    previewArea.innerHTML = '';
    

    uploadInterval = setInterval(() => {
        const currentWidth = parseFloat(progressBar.style.width) || 0;
        if (currentWidth < 100) {
            const newWidth = currentWidth + 10;
            progressBar.style.width = newWidth + '%';
            progressText.textContent = newWidth + '%';
        } else {
            clearInterval(uploadInterval);
            checkIcon.style.display = 'block';
            displayImage(file);
            dragDropArea.style.display = 'none';
            checkButton.style.display = 'block';
            or.style.display = 'none';
            
        }
    }, 200);
}

document.getElementById('file-input').addEventListener('change', function() {
    var label = document.getElementById('upload-label').querySelector('span');
    if (this.files.length > 0) {
        label.textContent = 'Upload Another File';
        
    } else {
        label.textContent = 'Upload File';
        testt.style.display = 'none';
    }
});

function displayImage(file) {
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = document.createElement('img');
        img.src = event.target.result;
        previewArea.appendChild(img);
        imageDataInput.value = event.target.result;
        uploadForm.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

checkButton.addEventListener('click', () => {
    if (fileInput.files.length === 0) {
        message.style.display = 'block';
    } else {
        message.style.display = 'none';
        uploadForm.submit();
    }
});
