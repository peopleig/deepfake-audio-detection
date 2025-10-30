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


