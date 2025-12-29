// scripts/train_gastrectomy_model.js
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs').promises;
const path = require('path');

// Импорт модулей
const DataPreprocessor = require('./utils/data_preprocessor');
const ModelArchitecture = require('./utils/model_architecture');
const MetricsCalculator = require('./metrics/calculate_metrics');
const { loadData } = require('./utils/data_loader');
const { validateDataStructure } = require('./validation/data_validator');
const config = require('./config/model_config');

async function main() {
  try {
    console.log('--- Starting Training Pipeline ---');

    // 1. Load Data
    console.log(`Loading data from ${config.DATA_PATH}`);
    const rawData = await loadData(config.DATA_PATH);

    // 2. Validate Data Structure
    console.log('Validating data structure...');
    validateDataStructure(rawData);

    // 3. Initialize Preprocessor
    console.log('Initializing preprocessor...');
    const preprocessor = new DataPreprocessor(config.FEATURE_COLUMNS, config.TARGET_COLUMN);

    // 4. Process Data
    console.log('Preprocessing data...');
    const { X, y, scaler } = await preprocessor.process(rawData);

    // 5. Create Model
    console.log('Creating model...');
    const model = ModelArchitecture.createModel(config.MODEL_ARCHITECTURE);

    // 6. Train Model
    console.log('Starting training...');
    const history = await model.fit(X, y, {
      epochs: config.TRAINING.epochs,
      batchSize: config.TRAINING.batchSize,
      validationSplit: config.TRAINING.validationSplit,
      verbose: 1
    });

    console.log('Training completed.');

    // 7. Evaluate Model (на X, y - в реальности нужна тестовая выборка)
    console.log('Calculating metrics...');
    const metrics = await MetricsCalculator.calculateBinaryClassificationMetrics(y, model.predict(X));

    // 8. Save Artifacts
    // Сохранение модели
    const modelSavePath = path.resolve(config.MODEL_SAVE_PATH.replace('.json', ''));
    console.log(`Saving model to ${modelSavePath}`);
    await model.save(`file://${modelSavePath}`);
    console.log('Model saved successfully.');

    // Сохранение scaler
    console.log(`Saving scaler to ${config.SCALER_SAVE_PATH}`);
    await preprocessor.saveScaler(config.SCALER_SAVE_PATH);

    // Сохранение метрик
    const metricsJson = JSON.stringify(metrics, null, 2);
    console.log(`Saving metrics to ${config.METRICS_SAVE_PATH}`);
    await fs.writeFile(config.METRICS_SAVE_PATH, metricsJson);
    console.log('Metrics saved successfully.');

    console.log('--- Training Pipeline Completed Successfully ---');

  } catch (error) {
    console.error('An error occurred during the training pipeline:', error);
    process.exit(1);
  } finally {
    // Очистка памяти
    tf.disposeVariables();
    console.log('Memory cleared.');
  }
}

if (require.main === module) {
  main();
}

module.exports = { main };
