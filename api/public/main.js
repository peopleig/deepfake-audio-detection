document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById("audioFile");
    const resultDiv = document.getElementById("result");
    const verdictEl = document.getElementById("verdict");
    const probEl = document.getElementById("probability");
    const labelText = document.getElementById("labelText");

    if (!fileInput.files.length) {
        alert("Please select an audio file.");
        return;
    }

    formData.append("audio", fileInput.files[0]);
    verdictEl.textContent = "Analyzing...";
    probEl.textContent = "";
    resultDiv.classList.remove("hidden");

    try {
        const res = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        verdictEl.textContent = `Verdict: ${data.verdict.toUpperCase()}`;
        probEl.textContent = `Confidence: ${(data.probability * 100).toFixed(2)}%`;
        
        // ✅ Show which model was used
        if (data.model_used) {
            document.getElementById('modelUsed').textContent = `(Using ${data.model_used})`;
        }
    } catch (err) {
        verdictEl.textContent = "Error analyzing file.";
        console.error(err);
    } finally {
        fileInput.value = "";
        labelText.textContent = "Choose File";
        document.getElementById("uploadForm").reset();
    }
});


const audioInput = document.getElementById('audioFile');
const labelText = document.getElementById('labelText');

audioInput.addEventListener('change', () => {
    if (audioInput.files && audioInput.files.length > 0) {
        labelText.textContent = `Uploaded: ${audioInput.files[0].name}`;
    } else {
        labelText.textContent = 'Choose File';
    }
});


let mediaRecorder;
let audioChunks = [];

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const fileInput = document.getElementById("audioFile");

recordBtn.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

        mediaRecorder.onstop = async () => {
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            const file = new File([blob], "recording.webm", { type: "audio/webm" });

            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;

            labelText.textContent = "Recorded Audio Ready ✅";
        };

        mediaRecorder.start();
        recordBtn.classList.add("hidden");
        stopBtn.classList.remove("hidden");
        labelText.textContent = "Recording...";
    } catch (err) {
        alert("Microphone access denied or not available.");
        console.error(err);
    }
});

stopBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        stopBtn.classList.add("hidden");
        recordBtn.classList.remove("hidden");
        labelText.textContent = "Processing...";
    }
});


// ✅ Auto-refresh training metrics every 30 seconds
async function refreshMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const metrics = await res.json();
        
        // Update DOM elements
        const metricValues = document.querySelectorAll('.metric-value');
        if (metricValues.length >= 6) {
            metricValues[0].textContent = metrics.cumulative_epochs || 0;
            metricValues[1].textContent = metrics.train_accuracy ? metrics.train_accuracy.toFixed(2) + '%' : 'N/A';
            metricValues[2].textContent = metrics.val_accuracy ? metrics.val_accuracy.toFixed(2) + '%' : 'N/A';
            metricValues[3].textContent = metrics.train_loss ? metrics.train_loss.toFixed(4) : 'N/A';
            metricValues[4].textContent = metrics.val_loss ? metrics.val_loss.toFixed(4) : 'N/A';
            metricValues[5].textContent = metrics.best_val_loss ? metrics.best_val_loss.toFixed(4) : 'N/A';
        }
        
        const lastUpdated = document.querySelector('.last-updated');
        if (lastUpdated && metrics.timestamp) {
            lastUpdated.textContent = `Last updated: ${metrics.timestamp}`;
        }
    } catch (err) {
        console.error('Failed to refresh metrics:', err);
    }
}

// Refresh every 30 seconds
setInterval(refreshMetrics, 30000);


// ✅ Model Toggle Functionality

// Load current model on page load
async function loadCurrentModel() {
    try {
        const res = await fetch('/api/current-model');
        const data = await res.json();
        const modelNameEl = document.getElementById('modelName');
        if (modelNameEl) {
            modelNameEl.textContent = `Model: ${data.model}`;
        }
    } catch (err) {
        console.error('Failed to load current model:', err);
    }
}

// Toggle model when button is clicked
const toggleModelBtn = document.getElementById('toggleModelBtn');
if (toggleModelBtn) {
    toggleModelBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/toggle-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await res.json();
            
            const modelNameEl = document.getElementById('modelName');
            if (modelNameEl) {
                modelNameEl.textContent = `Model: ${data.model}`;
            }
            
            // alert(`Switched to ${data.model}`);
        } catch (err) {
            console.error('Failed to toggle model:', err);
            // alert('Failed to switch model');
        }
    });
}

// Load current model when page loads
loadCurrentModel();
