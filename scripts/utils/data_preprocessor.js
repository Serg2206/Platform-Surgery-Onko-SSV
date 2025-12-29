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
    const colsToEncode = categoricalCols.filter(col => df.columnNames.includes(col));
    
    if (colsToEncode.length > 0) {
      df = df.oneHotEncode({ column: colsToEncode, prefix: colsToEncode });
    }

    // 2. Выделение X и y
    const X = df.loc({ columns: this.featureColumns });
    const y = df.loc({ columns: [this.targetColumn] });

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
    const y_tensor = y.tensor.cast('bool').cast('float32');

    console.log(`Preprocessing complete. X shape: [${X_tensor.shape}], y shape: [${y_tensor.shape}]`);
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
      const scalerJson = JSON.stringify(this.scaler.toJSON());
      const fs = require('fs').promises;
      await fs.writeFile(path, scalerJson);
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
    const scalerJson = JSON.parse(jsonStr);
    const scaler = new StandardScaler();
    scaler.fromJSON(scalerJson);
    return new DataPreprocessor(null, null, scaler);
  }
}

module.exports = DataPreprocessor;
