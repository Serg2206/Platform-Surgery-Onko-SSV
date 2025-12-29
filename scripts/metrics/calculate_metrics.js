// scripts/metrics/calculate_metrics.js
const tf = require('@tensorflow/tfjs-node');

/**
 * MetricsCalculator - Класс для вычисления метрик модели
 * Поддерживает бинарную классификацию и регрессию
 */
class MetricsCalculator {
  /**
   * Вычисление метрик для бинарной классификации.
   * @param {tf.Tensor} y_true - Истинные метки.
   * @param {tf.Tensor} y_pred - Предсказанные вероятности.
   * @returns {Object} - Метрики (AUC, accuracy, precision, recall, F1, confusion matrix).
   */
  static async calculateBinaryClassificationMetrics(y_true, y_pred) {
    // Бинаризация предсказаний (порог 0.5)
    const y_pred_binary = y_pred.greaterEqual(tf.scalar(0.5));

    // Вычисление confusion matrix
    const y_true_bool = y_true.cast('bool');
    const y_pred_bool = y_pred_binary.cast('bool');

    const tp = y_true_bool.logicalAnd(y_pred_bool).sum().dataSync()[0];
    const fp = y_true_bool.logicalNot().logicalAnd(y_pred_bool).sum().dataSync()[0];
    const tn = y_true_bool.logicalNot().logicalAnd(y_pred_bool.logicalNot()).sum().dataSync()[0];
    const fn = y_true_bool.logicalAnd(y_pred_bool.logicalNot()).sum().dataSync()[0];

    // Метрики
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp / (tp + fp) || 0; // Избегаем деления на 0
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;

    // AUC (используем Promise для async)
    const auc = await tf.metrics.auc(y_true, y_pred).data();

    console.log(`Metrics - AUC: ${auc[0].toFixed(4)}, Accuracy: ${accuracy.toFixed(4)}, Precision: ${precision.toFixed(4)}, Recall: ${recall.toFixed(4)}, F1: ${f1.toFixed(4)}`);

    const metrics = {
      auc: auc[0],
      accuracy: accuracy,
      precision: precision,
      recall: recall,
      f1: f1,
      confusion_matrix: { tp, fp, tn, fn }
    };

    // Очистка памяти
    y_pred_binary.dispose();
    y_true_bool.dispose();
    y_pred_bool.dispose();

    return metrics;
  }

  /**
   * Вычисление метрик для регрессии, если понадобится.
   * @param {tf.Tensor} y_true - Истинные значения.
   * @param {tf.Tensor} y_pred - Предсказанные значения.
   * @returns {Object} - Метрики (MSE, RMSE, MAE).
   */
  static calculateRegressionMetrics(y_true, y_pred) {
    const mse = y_true.squaredDifference(y_pred).mean().dataSync()[0];
    const rmse = Math.sqrt(mse);
    const mae = y_true.sub(y_pred).abs().mean().dataSync()[0];

    console.log(`Regression Metrics - MSE: ${mse.toFixed(4)}, RMSE: ${rmse.toFixed(4)}, MAE: ${mae.toFixed(4)}`);

    return {
      mse: mse,
      rmse: rmse,
      mae: mae
    };
  }
}

module.exports = MetricsCalculator;
