const express = require('express');
const fs = require('fs');
const path = require('path');
const PORT = 3000;
const app = express();

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// ✅ Store current model selection in memory
let currentModel = 'for_resnet34-1';  // Default model

// ✅ Export BEFORE requiring analyze_router
module.exports = { getCurrentModel: () => currentModel };

// ✅ NOW require the router (after exporting)
const analyze_router = require('./routes/analyze.js');

// Function to read training metrics
function getTrainingMetrics() {
    try {
        const metricsPath = path.join(__dirname, '..', 'latest_metrics.json');
        
        if (fs.existsSync(metricsPath)) {
            const data = fs.readFileSync(metricsPath, 'utf8');
            return JSON.parse(data);
        } else {
            return getDefaultMetrics();
        }
    } catch (error) {
        console.error('Error reading metrics:', error);
        return getDefaultMetrics();
    }
}

function getEpochCounter() {
    try {
        const counterPath = path.join(__dirname, '..', 'epoch_counter.json');
        
        if (fs.existsSync(counterPath)) {
            const data = fs.readFileSync(counterPath, 'utf8');
            const parsed = JSON.parse(data);
            return parsed.total_epochs || 0;
        } else {
            return 0;
        }
    } catch (error) {
        console.error('Error reading epoch_counter.json:', error);
        return 0;
    }
}

function getDefaultMetrics() {
    return {
        cumulative_epochs: 0,
        train_accuracy: null,
        val_accuracy: null,
        train_loss: null,
        val_loss: null,
        best_val_loss: null,
        timestamp: null
    };
}

// Main route
app.get('/', (req, res) => {
    const metrics = getTrainingMetrics();
    const totalEpochs = getEpochCounter();
    metrics.cumulative_epochs = totalEpochs;
    
    res.render('homes', { metrics, currentModel });
});

// API endpoint to get current model
app.get('/api/current-model', (req, res) => {
    res.json({ model: currentModel });
});

// API endpoint to toggle model
app.post('/api/toggle-model', (req, res) => {
    if (currentModel === 'for_resnet34-1') {
        currentModel = 'for_resnet34-2';
    } else {
        currentModel = 'for_resnet34-1';
    }
    console.log(`Model switched to: ${currentModel}`);
    res.json({ model: currentModel });
});

app.get('/api/metrics', (req, res) => {
    const metrics = getTrainingMetrics();
    const totalEpochs = getEpochCounter();
    metrics.cumulative_epochs = totalEpochs;
    res.json(metrics);
});

app.use("/analyze", analyze_router);

app.listen(PORT, () => {
    console.log(`Server active on http://localhost:${PORT}`);
    console.log(`Metrics path: ${path.join(__dirname, '..', 'latest_metrics.json')}`);
    console.log(`Counter path: ${path.join(__dirname, '..', 'epoch_counter.json')}`);
});
