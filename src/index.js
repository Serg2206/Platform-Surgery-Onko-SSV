/**
 * Surgery-Onko-SSV Platform
 * Main entry point for molecular data analysis
 * ⚠️ DEMO VERSION: Uses synthetic data only
 */
const express = require('express');
const path = require('path');
const helmet = require('helmet');
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

// Input validation helper
const validatePredictionInput = (data) => {
  const required = ['age', 'sex', 'bmi', 'tumor_stage', 'surgery_type'];
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
app.post('/api/v1/predict', authenticateToken, (req, res) => {
  const validation = validatePredictionInput(req.body);
  
  if (!validation.valid) {
    return res.status(400).json({
      error: 'Validation Error',
      message: validation.error
    });
  }
  
  // Placeholder: In production, load model and run prediction
  res.status(200).json({
    message: 'Prediction endpoint (DEMO)',
    warning: 'This is a demonstration. Predictions are not based on real clinical data.',
    input: req.body,
    prediction: {
      complicationRisk: 0.15,
      confidence: 0.72,
      disclaimer: 'NOT FOR CLINICAL USE'
    }
  });
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
