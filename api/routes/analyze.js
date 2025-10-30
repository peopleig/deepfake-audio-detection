const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

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

router.post("/", upload.single("audio"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No audio file uploaded" });
        }

        const filePath = path.join(__dirname, "../uploads", req.file.filename);
        const projectRoot = path.resolve(__dirname, "../../");
        const pythonScript = path.join(projectRoot, "src", "predict.py");

        const python = spawn("python3", [
            pythonScript,
            "--audio", filePath,
            "--ckpt", path.join(projectRoot, "runs", "for_resnet34.pt")
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
            fs.unlink(filePath, (err) => {
                if (err) console.error("Error deleting uploaded file:", err);
            });
            if (code !== 0) {
                console.error("Python exited with error:", errorOutput);
                return res.status(500).json({ error: "Model prediction failed" });
            }
            try {
                const result = JSON.parse(output);
                res.json(result);
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
