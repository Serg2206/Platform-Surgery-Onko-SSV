// scripts/utils/data_preprocessor.js
const dfd = require('danfojs-node');
const { StandardScaler } = require('danfojs-node');

/**
 * DataPreprocessor - Класс для предобработки данных
 * - One-hot encoding категориальных признаков
 * - Нормализация StandardScaler
 * - Сохранение/загрузка scaler для inference
 */
class DataPreprocessor {
  constructor(featureColumns, targetColumn, scaler = null) {
    this.featureColumns = featureColumns;
    this.targetColumn = targetColumn;
    this.scaler = scaler; // scaler для inference
    this.fitted = !!scaler; // scaler уже подготовлен?
  }

  /**
   * Основной метод: one-hot кодирование + нормализация.
   * @param {Array<Object>} rawData - JSON массив или DataFrame.
   * @returns {Object} {X:tensor, y:tensor, scaler}.
   */
  async process(rawData) {
    let df = rawData instanceof dfd.DataFrame ? rawData : new dfd.DataFrame(rawData);

    // 1. One-hot encoding: sex, tumor_stage, surgery_type, neoadjuvant_therapy
    const categoricalCols = ['sex', 'tumor_stage', 'surgery_type', 'neoadjuvant_therapy'];
    const currentColumns = df.columns;
    const colsToEncode = categoricalCols.filter(col => currentColumns.includes(col));
    
    if (colsToEncode.length > 0) {
      df = dfd.getDummies(df, { columns: colsToEncode });
    }

    // Добавляем недостающие колонки (важно для inference)
    if (this.featureColumns && this.fitted) {
      for (const col of this.featureColumns) {
        if (!df.columns.includes(col)) {
          df = df.addColumn(col, new Array(df.shape[0]).fill(0));
        }
      }
    }

    // 2. Выделение X и y
    // Если featureColumns не заданы (inference с уже обученным scaler), используем все колонки
    const X_cols = this.featureColumns || df.columns;
    const X = df.loc({ columns: X_cols });

    let y = null;
    if (this.targetColumn && df.columns.includes(this.targetColumn)) {
      y = df.loc({ columns: [this.targetColumn] });
    }

    // 3. Нормализация StandardScaler
    let X_scaled;
    if (!this.fitted) {
      // Обучение: fit + transform
      this.scaler = new StandardScaler();
      await this.scaler.fit(X);
      X_scaled = this.scaler.transform(X);
      this.fitted = true;
    } else {
      // Inference: только transform
      X_scaled = this.scaler.transform(X);
    }

    // 4. Преобразование в тензоры tf
    const X_tensor = X_scaled.tensor.asType('float32');
    let y_tensor = null;
    if (y) {
      y_tensor = y.tensor.cast('bool').cast('float32');
    }

    console.log(`Preprocessing complete. X shape: [${X_tensor.shape}]${y_tensor ? `, y shape: [${y_tensor.shape}]` : ''}`);
    console.log(`Fitted scaler: ${this.fitted}`);

    return {
      X: X_tensor,
      y: y_tensor,
      scaler: this.scaler
    };
  }

  /**
   * Сохранение scaler в JSON.
   */
  async saveScaler(path) {
    if (this.scaler && this.fitted) {
      // В новых версиях danfojs-node метод toJSON() может отсутствовать у StandardScaler
      // Сохраняем параметры вручную
      const params = {
        mean: this.scaler.$mean.arraySync(),
        std: this.scaler.$std.arraySync()
      };
      const fs = require('fs').promises;
      await fs.writeFile(path, JSON.stringify(params));
      console.log(`Scaler saved to ${path}`);
    } else {
      console.error('Cannot save scaler: not fitted yet.');
    }
  }

  /**
   * Загрузка scaler из JSON.
   */
  static async loadScaler(path) {
    const fs = require('fs').promises;
    const jsonStr = await fs.readFile(path, 'utf8');
    const params = JSON.parse(jsonStr);
    const scaler = new StandardScaler();

    // Восстанавливаем среднее и стандартное отклонение из нашего формата {mean, std}
    if (params.mean && params.std) {
      const tf = require('@tensorflow/tfjs-node');
      // В среде Jest require может вызывать ошибки при асинхронной инициализации
      if (tf.tensor1d) {
        scaler.$mean = tf.tensor1d(params.mean);
        scaler.$std = tf.tensor1d(params.std);
      }
    } else {
      scaler.fromJSON(params);
    }

    const config = require('../config/model_config');
    return new DataPreprocessor(config.FEATURE_COLUMNS, config.TARGET_COLUMN, scaler);
  }
}

module.exports = DataPreprocessor;
