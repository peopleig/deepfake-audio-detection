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

