const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const { spawn } = require("child_process");

const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, path.join(__dirname, "../uploads")),
    filename: (req, file, cb) => cb(null, file.originalname)
});
const upload = multer({ storage });

router.post("/", upload.single("audio"), async (req, res) => {
    try {
        const projectRoot = path.resolve(__dirname, "../../");
        const dummyScript = path.join(projectRoot, "src", "dummy.py");

        const python = spawn("python3", [dummyScript]);

        let output = "";
        let errorOutput = "";

        python.stdout.on("data", (data) => {
            output += data.toString();
        });

        python.stderr.on("data", (data) => {
            errorOutput += data.toString();
        });

        python.on("close", (code) => {
            if (code !== 0) {
                console.error("Dummy script error:", errorOutput);
                return res.status(500).json({ error: "Dummy model failed" });
            }

            try {
                const result = JSON.parse(output);
                res.json(result);
            } catch (err) {
                console.error("Invalid JSON from dummy:", output);
                res.status(500).json({ error: "Invalid dummy output" });
            }
        });
    } catch (err) {
        console.error("Error in /analyze dummy route:", err);
        res.status(500).json({ error: "Server error" });
    }
});

module.exports = router;
