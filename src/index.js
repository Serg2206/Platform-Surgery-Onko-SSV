/**
 * Surgery-Onko-SSV Platform
 * Main entry point for molecular data analysis
 * ⚠️ DEMO VERSION: Uses synthetic data only
 */
const express = require('express');
const path = require('path');
const helmet = require('helmet');
const tf = require('@tensorflow/tfjs-node');
const DataPreprocessor = require('../scripts/utils/data_preprocessor');
const ModelArchitecture = require('../scripts/utils/model_architecture');
const config = require('../scripts/config/model_config');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Security middleware
app.use(helmet());

// Middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Simple authentication middleware (Bearer token)
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (!token || token !== process.env.API_TOKEN) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Access token is required or invalid'
    });
  }

  next();
};

// Model state
let model = null;
let preprocessor = null;

// Load model and scaler
async function initModel() {
  try {
    console.log('Initializing ML model...');
    const modelPath = path.resolve(__dirname, '..', config.MODEL_SAVE_PATH);
    const scalerPath = path.resolve(__dirname, '..', config.SCALER_SAVE_PATH);

    // Check if files exist
    const fs = require('fs').promises;
    await fs.access(path.join(modelPath, 'model.json'));
    await fs.access(scalerPath);

    model = await ModelArchitecture.loadModel(path.join(modelPath, 'model.json'));
    preprocessor = await DataPreprocessor.loadScaler(scalerPath);

    console.log('✓ ML model and scaler loaded successfully');
  } catch (err) {
    console.warn('⚠️  Could not load ML model. API will run in DEMO mode with dummy data.');
    console.warn(`Details: ${err.message}`);
  }
}

// Initialize model only if not in test environment or if explicitly requested
if (process.env.NODE_ENV !== 'test') {
  initModel();
}

// Export for testing
app.initModel = initModel;

// Input validation helper
const validatePredictionInput = (data) => {
  const required = [
    'age', 'sex', 'bmi', 'tumor_stage', 'surgery_type',
    'operation_time_min', 'blood_loss_ml', 'lymph_nodes_removed', 'neoadjuvant_therapy'
  ];
  const missing = required.filter(field => !(field in data));
  
  if (missing.length > 0) {
    return { valid: false, error: `Missing required fields: ${missing.join(', ')}` };
  }
  
  if (typeof data.age !== 'number' || data.age < 0 || data.age > 120) {
    return { valid: false, error: 'Age must be a number between 0 and 120' };
  }
  
  if (!['M', 'F'].includes(data.sex)) {
    return { valid: false, error: 'Sex must be "M" or "F"' };
  }
  
  if (typeof data.bmi !== 'number' || data.bmi < 10 || data.bmi > 60) {
    return { valid: false, error: 'BMI must be a number between 10 and 60' };
  }
  
  return { valid: true };
};

// Health check endpoint (public)
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    service: 'Surgery-Onko-SSV Platform',
    version: '1.0.0',
    mode: 'DEMO (synthetic data)',
    timestamp: new Date().toISOString()
  });
});

// API routes (protected)
app.get('/api/v1/info', authenticateToken, (req, res) => {
  res.json({
    name: 'Platform-Surgery-Onko-SSV',
    description: 'Платформа для анализа молекулярных данных в контексте хирургии и онкологии',
    mode: 'DEMO',
    dataStatus: 'Synthetic data only - Not for clinical use',
    features: [
      'Анализ молекулярных данных',
      'Предобработка данных',
      'Работа с обученными моделями ML'
    ],
    technologies: ['TensorFlow.js', 'Express', 'danfojs-node', 'helmet']
  });
});

// Protected ML prediction endpoint with validation
app.post('/api/v1/predict', authenticateToken, async (req, res) => {
  try {
    const validation = validatePredictionInput(req.body);

    if (!validation.valid) {
      return res.status(400).json({
        error: 'Validation Error',
        message: validation.error
      });
    }

    if (!model || !preprocessor) {
      return res.status(200).json({
        message: 'Prediction endpoint (DEMO - Model not loaded)',
        warning: 'This is a fallback demonstration. Model is not available.',
        input: req.body,
        prediction: {
          complicationRisk: 0.25,
          confidence: 0.5,
          mode: 'MOCK',
          disclaimer: 'NOT FOR CLINICAL USE'
        }
      });
    }

    // Preprocess input
    const { X } = await preprocessor.process([req.body]);

    // Predict
    const predictionTensor = model.predict(X);
    const risk = (await predictionTensor.data())[0];

    // Response
    res.status(200).json({
      message: 'Gastrectomy complication risk prediction',
      status: 'success',
      input: req.body,
      prediction: {
        complicationRisk: parseFloat(risk.toFixed(4)),
        hasRisk: risk >= 0.5,
        confidence: 0.85, // В реальности можно брать из неопределенности модели
        disclaimer: 'NOT FOR CLINICAL USE - RESEARCH ONLY'
      }
    });

    // Cleanup
    X.dispose();
    predictionTensor.dispose();

  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({
      error: 'Prediction Failed',
      message: error.message
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err.stack);
  res.status(500).json({
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.url} not found`
  });
});

// Start server
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`\n🔒 Helmet security middleware enabled`);
    console.log(`🔐 Authentication enabled (API_TOKEN required)`);
    console.log(`⚠️  DEMO MODE: Using synthetic data only`);
    console.log(`🚀 Surgery-Onko-SSV Platform`);
    console.log(`📊 Server running on port ${PORT}`);
    console.log(`🌐 Health check: http://localhost:${PORT}/health`);
    console.log(`📖 API info (protected): http://localhost:${PORT}/api/v1/info`);
    console.log(`🔮 API predict (protected): http://localhost:${PORT}/api/v1/predict`);
    console.log(`💡 Set API_TOKEN in .env for protected routes\n`);
  });
}

module.exports = app;
