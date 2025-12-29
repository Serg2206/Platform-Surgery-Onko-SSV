// scripts/utils/model_architecture.js
const tf = require('@tensorflow/tfjs-node');

/**
 * ModelArchitecture - Класс для создания архитектуры нейронной сети
 * Использует конфигурацию из model_config.js
 */
class ModelArchitecture {
  /**
   * Создает модель на основе конфигурации.
   * @param {Object} config - Конфиг с inputSize, layers.
   * @returns {tf.Sequential} - Скомпилированная модель.
   */
  static createModel(config) {
    const model = tf.sequential();

    // Добавляем первый слой с inputShape
    model.add(tf.layers.dense({
      units: config.layers[0].units,
      activation: config.layers[0].activation,
      inputShape: [config.inputSize] // inputSize из конфига
    }));

    // Добавляем dropout после первого слоя
    if (config.layers[0].dropout) {
      model.add(tf.layers.dropout({ rate: config.layers[0].dropout }));
    }

    // Добавляем остальные скрытые слои
    for (let i = 1; i < config.layers.length - 1; i++) {
      model.add(tf.layers.dense({
        units: config.layers[i].units,
        activation: config.layers[i].activation
      }));

      // Добавляем dropout
      if (config.layers[i].dropout) {
        model.add(tf.layers.dropout({ rate: config.layers[i].dropout }));
      }
    }

    // Добавляем выходной слой
    model.add(tf.layers.dense({
      units: config.layers[config.layers.length - 1].units,
      activation: config.layers[config.layers.length - 1].activation
    }));

    // Компилируем модель
    model.compile({
      optimizer: config.optimizer || 'adam',
      loss: config.loss || 'binaryCrossentropy',
      metrics: config.metrics || ['accuracy']
    });

    console.log('Model architecture created and compiled successfully.');
    return model;
  }

  /**
   * Загрузка сохраненной модели.
   * @param {string} path - Путь к модели, например file://./models/saved_model.
   * @returns {tf.Sequential} - Загруженная модель.
   */
  static async loadModel(path) {
    const model = await tf.loadLayersModel(`file://${path}`);
    console.log(`Model loaded from ${path}`);
    return model;
  }
}

module.exports = ModelArchitecture;
