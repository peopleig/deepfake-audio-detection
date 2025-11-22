const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const ffmpegPath = require("ffmpeg-static");
const ffmpeg = require("fluent-ffmpeg");

// ✅ Import getCurrentModel function from app.js
const { getCurrentModel } = require('../app.js');

// Tell fluent-ffmpeg where FFmpeg binary is
ffmpeg.setFfmpegPath(ffmpegPath);

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, "../uploads");
        if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + "-" + Math.floor(Math.random() * 1e4);
        const ext = path.extname(file.originalname);
        cb(null, file.fieldname + "-" + uniqueSuffix + ext);
    }
});
const upload = multer({ storage });

// Helper function to convert any audio to WAV
function convertToWav(inputPath, outputPath) {
    return new Promise((resolve, reject) => {
        ffmpeg(inputPath)
            .toFormat('wav')
            .audioChannels(1)  // Mono
            .audioFrequency(16000)  // Sample rate
            .on('end', () => {
                console.log('Conversion complete');
                resolve(outputPath);
            })
            .on('error', (err) => {
                console.error('Conversion error:', err);
                reject(err);
            })
            .save(outputPath);
    });
}

router.post("/", upload.single("audio"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No audio file uploaded" });
        }

        const uploadedFilePath = path.join(__dirname, "../uploads", req.file.filename);
        const projectRoot = path.resolve(__dirname, "../../");
        const pythonScript = path.join(projectRoot, "src", "predict.py");
        const venvPython = path.join(projectRoot, "deepfake-env", "Scripts", "python.exe");

        // Convert to WAV if not already
        let audioFilePath = uploadedFilePath;
        const ext = path.extname(req.file.filename).toLowerCase();
        
        if (ext !== ".wav") {
            const wavPath = uploadedFilePath.replace(ext, ".wav");
            try {
                await convertToWav(uploadedFilePath, wavPath);
                audioFilePath = wavPath;
                // Delete original file after conversion
                fs.unlink(uploadedFilePath, (err) => {
                    if (err) console.error("Error deleting original file:", err);
                });
            } catch (err) {
                console.error("Audio conversion failed:", err);
                fs.unlink(uploadedFilePath, () => {});
                return res.status(500).json({ error: "Audio conversion failed" });
            }
        }

        // ✅ Get current model dynamically
        const currentModel = getCurrentModel();
        const checkpointPath = path.join(projectRoot, "runs", `${currentModel}.pt`);
        
        console.log(`🔍 Using model: ${currentModel}`);

        const python = spawn(venvPython, [
            pythonScript,
            "--audio", audioFilePath,
            "--ckpt", checkpointPath  // ✅ Use dynamic checkpoint path
        ]);

        let output = "";
        let errorOutput = "";

        python.stdout.on("data", (data) => {
            output += data.toString();
        });

        python.stderr.on("data", (data) => {
            errorOutput += data.toString();
        });

        python.on("close", (code) => {
            // Clean up WAV file
            fs.unlink(audioFilePath, (err) => {
                if (err) console.error("Error deleting audio file:", err);
            });
            
            if (code !== 0) {
                console.error("Python exited with error:", errorOutput);
                return res.status(500).json({ error: "Model prediction failed" });
            }
            
            try {
                const result = JSON.parse(output);
                
                // Transform probability to confidence in the predicted class
                const deepfakeProb = result.probability;
                let verdict, confidence;
                
                if (deepfakeProb >= 0.5) {
                    verdict = "deepfake";
                    confidence = deepfakeProb;
                } else {
                    verdict = "real";
                    confidence = 1 - deepfakeProb;
                }
                
                // Send user-friendly response
                res.json({
                    verdict: verdict,
                    probability: confidence,
                    confidence_percent: (confidence * 100).toFixed(2),
                    rating: confidence >= 0.75 ? "High Confidence" : confidence >= 0.5 ? "Medium Confidence" : "Low Confidence",
                    model_used: currentModel  // ✅ Return which model was used
                });
                
            } catch (err) {
                console.error("Invalid JSON from Python:", output);
                res.status(500).json({ error: "Invalid model output" });
            }
        });

    } catch (err) {
        console.error("Error in /analyze:", err);
        res.status(500).json({ error: "Server error" });
    }
});

module.exports = router;
