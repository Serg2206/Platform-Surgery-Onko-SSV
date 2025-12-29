/**
 * Surgery-Onko-SSV Platform
 * Main entry point for molecular data analysis
 */

const express = require('express');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    service: 'Surgery-Onko-SSV Platform',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
});

// API routes
app.get('/api/v1/info', (req, res) => {
  res.json({
    name: 'Platform-Surgery-Onko-SSV',
    description: 'Платформа для анализа молекулярных данных в контексте хирургии и онкологии',
    features: [
      'Анализ молекулярных данных',
      'Предобработка данных',
      'Работа с обученными моделями ML'
    ],
    technologies: ['TensorFlow.js', 'Express', 'danfojs-node']
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
    console.log(`\n🚀 Surgery-Onko-SSV Platform`);
    console.log(`📊 Server running on port ${PORT}`);
    console.log(`🌐 Health check: http://localhost:${PORT}/health`);
    console.log(`📖 API info: http://localhost:${PORT}/api/v1/info\n`);
  });
}

module.exports = app;
