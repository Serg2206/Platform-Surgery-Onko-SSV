/**
 * K-Fold Cross-Validation for Gastrectomy Risk Model
 * PR #5: Реализация строгой валидации модели
 * 
 * Функциональность:
 * - K-fold стратифицированная кросс-валидация
 * - Расчет агрегированных метрик (среднее ± std)
 * - Генерация подробного отчета
 * - Визуализация результатов каждого фолда
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs').promises;
const { loadAndPreprocessData, createModel, trainModel } = require('./train_gastrectomy_model');

// Конфигурация кросс-валидации
const CV_CONFIG = {
  nFolds: 5,
  stratified: true,
  randomSeed: 42,
  resultsPath: './results/cv_gastrectomy'
};

/**
 * Разделение данных на K фолдов со стратификацией
 */
function createStratifiedFolds(data, labels, k) {
  console.log(`\n📊 Creating ${k} stratified folds...`);
  
  // Группировка индексов по классам
  const positiveIndices = [];
  const negativeIndices = [];
  
  labels.forEach((label, idx) => {
    if (label === 1) positiveIndices.push(idx);
    else negativeIndices.push(idx);
  });
  
  // Shuffle с фиксированным seed
  const shuffle = (arr, seed) => {
    const rng = (s) => {
      s = Math.sin(s) * 10000;
      return s - Math.floor(s);
    };
    let currentSeed = seed;
    return arr.sort(() => {
      currentSeed++;
      return rng(currentSeed) - 0.5;
    });
  };
  
  shuffle(positiveIndices, CV_CONFIG.randomSeed);
  shuffle(negativeIndices, CV_CONFIG.randomSeed + 1);
  
  // Разделение на фолды
  const folds = [];
  const posPerFold = Math.floor(positiveIndices.length / k);
  const negPerFold = Math.floor(negativeIndices.length / k);
  
  for (let i = 0; i < k; i++) {
    const posStart = i * posPerFold;
    const negStart = i * negPerFold;
    
    const posEnd = i === k - 1 ? positiveIndices.length : (i + 1) * posPerFold;
    const negEnd = i === k - 1 ? negativeIndices.length : (i + 1) * negPerFold;
    
    const foldIndices = [
      ...positiveIndices.slice(posStart, posEnd),
      ...negativeIndices.slice(negStart, negEnd)
    ];
    
    folds.push({
      indices: foldIndices,
      positive: posEnd - posStart,
      negative: negEnd - negStart
    });
  }
  
  console.log('✓ Folds created:');
  folds.forEach((fold, i) => {
    console.log(
      `  Fold ${i + 1}: ${fold.indices.length} samples ` +
      `(${fold.positive} positive, ${fold.negative} negative)`
    );
  });
  
  return folds;
}

/**
 * Разделение тензоров на train/test по индексам
 */
function splitByIndices(xs, ys, trainIndices, testIndices) {
  const xsArray = xs.arraySync();
  const ysArray = ys.arraySync();
  
  const xTrain = trainIndices.map(i => xsArray[i]);
  const yTrain = trainIndices.map(i => ysArray[i]);
  const xTest = testIndices.map(i => xsArray[i]);
  const yTest = testIndices.map(i => ysArray[i]);
  
  return {
    xTrain: tf.tensor2d(xTrain),
    yTrain: tf.tensor2d(yTrain),
    xTest: tf.tensor2d(xTest),
    yTest: tf.tensor2d(yTest)
  };
}

/**
 * Оценка модели на тестовом фолде
 */
async function evaluateFold(model, xTest, yTest) {
  const predictions = await model.predict(xTest).array();
  const labels = await yTest.array();
  
  const threshold = 0.5;
  const binary = predictions.map(p => p[0] >= threshold ? 1 : 0);
  
  let tp = 0, fp = 0, tn = 0, fn = 0;
  
  binary.forEach((pred, i) => {
    const actual = labels[i][0];
    if (pred === 1 && actual === 1) tp++;
    if (pred === 1 && actual === 0) fp++;
    if (pred === 0 && actual === 0) tn++;
    if (pred === 0 && actual === 1) fn++;
  });
  
  const accuracy = (tp + tn) / (tp + tn + fp + fn);
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = 2 * (precision * recall) / (precision + recall) || 0;
  const specificity = tn / (tn + fp) || 0;
  
  // AUC-ROC
  const pairs = predictions.map((pred, i) => ({
    pred: pred[0],
    label: labels[i][0]
  })).sort((a, b) => b.pred - a.pred);
  
  let positives = 0, negatives = 0, sumRanks = 0;
  pairs.forEach((pair, i) => {
    if (pair.label === 1) {
      positives++;
      sumRanks += i + 1;
    } else {
      negatives++;
    }
  });
  
  const auc = (sumRanks - (positives * (positives + 1)) / 2) / (positives * negatives);
  
  return {
    accuracy,
    precision,
    recall,
    f1,
    specificity,
    auc,
    confusionMatrix: { tp, fp, tn, fn }
  };
}

/**
 * Запуск K-fold кросс-валидации
 */
async function runCrossValidation() {
  console.log('\n' + '='.repeat(70));
  console.log('🔬 K-FOLD CROSS-VALIDATION: GASTRECTOMY RISK MODEL');
  console.log('='.repeat(70));
  console.log(`\nConfiguration:`);
  console.log(`  K-folds: ${CV_CONFIG.nFolds}`);
  console.log(`  Stratified: ${CV_CONFIG.stratified}`);
  console.log(`  Random seed: ${CV_CONFIG.randomSeed}`);
  
  try {
    // 1. Загрузка данных
    const { xs, ys } = await loadAndPreprocessData();
    const labelsArray = await ys.arraySync();
    const flatLabels = labelsArray.map(l => l[0]);
    
    // 2. Создание фолдов
    const folds = createStratifiedFolds(
      await xs.arraySync(),
      flatLabels,
      CV_CONFIG.nFolds
    );
    
    // 3. Обучение и оценка на каждом фолде
    const foldResults = [];
    
    for (let foldIdx = 0; foldIdx < folds.length; foldIdx++) {
      console.log(`\n${'='.repeat(70)}`);
      console.log(`🎯 FOLD ${foldIdx + 1}/${CV_CONFIG.nFolds}`);
      console.log('='.repeat(70));
      
      const testIndices = folds[foldIdx].indices;
      const trainIndices = folds
        .filter((_, i) => i !== foldIdx)
        .flatMap(f => f.indices);
      
      console.log(`  Train samples: ${trainIndices.length}`);
      console.log(`  Test samples: ${testIndices.length}`);
      
      // Разделение данных
      const { xTrain, yTrain, xTest, yTest } = splitByIndices(
        xs, ys, trainIndices, testIndices
      );
      
      // Создание и обучение модели
      const model = createModel(xs.shape[1]);
      await trainModel(model, xTrain, yTrain);
      
      // Оценка
      const metrics = await evaluateFold(model, xTest, yTest);
      foldResults.push(metrics);
      
      console.log(`\n📊 Fold ${foldIdx + 1} Results:`);
      console.log(`  Accuracy:    ${metrics.accuracy.toFixed(4)}`);
      console.log(`  Precision:   ${metrics.precision.toFixed(4)}`);
      console.log(`  Recall:      ${metrics.recall.toFixed(4)}`);
      console.log(`  F1-Score:    ${metrics.f1.toFixed(4)}`);
      console.log(`  Specificity: ${metrics.specificity.toFixed(4)}`);
      console.log(`  AUC-ROC:     ${metrics.auc.toFixed(4)}`);
      
      // Очистка
      xTrain.dispose();
      yTrain.dispose();
      xTest.dispose();
      yTest.dispose();
      model.dispose();
    }
    
    // 4. Агрегация результатов
    console.log(`\n${'='.repeat(70)}`);
    console.log('📈 AGGREGATED RESULTS (Mean ± Std)');
    console.log('='.repeat(70) + '\n');
    
    const calculateStats = (metric) => {
      const values = foldResults.map(r => r[metric]);
      const mean = values.reduce((a, b) => a + b) / values.length;
      const std = Math.sqrt(
        values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length
      );
      return { mean, std, values };
    };
    
    const aggregated = {
      accuracy: calculateStats('accuracy'),
      precision: calculateStats('precision'),
      recall: calculateStats('recall'),
      f1: calculateStats('f1'),
      specificity: calculateStats('specificity'),
      auc: calculateStats('auc')
    };
    
    Object.entries(aggregated).forEach(([metric, stats]) => {
      console.log(
        `${metric.padEnd(15)}: ${stats.mean.toFixed(4)} ± ${stats.std.toFixed(4)}`
      );
    });
    
    console.log('\n📊 Per-Fold Breakdown:');
    foldResults.forEach((result, i) => {
      console.log(
        `  Fold ${i + 1}: ` +
        `Acc=${result.accuracy.toFixed(3)}, ` +
        `Prec=${result.precision.toFixed(3)}, ` +
        `Rec=${result.recall.toFixed(3)}, ` +
        `AUC=${result.auc.toFixed(3)}`
      );
    });
    
    // 5. Сохранение результатов
    const report = {
      config: CV_CONFIG,
      timestamp: new Date().toISOString(),
      folds: foldResults,
      aggregated: Object.entries(aggregated).reduce(
        (acc, [key, val]) => {
          acc[key] = { mean: val.mean, std: val.std };
          return acc;
        },
        {}
      )
    };
    
    await fs.mkdir(CV_CONFIG.resultsPath, { recursive: true });
    await fs.writeFile(
      `${CV_CONFIG.resultsPath}/cv_results.json`,
      JSON.stringify(report, null, 2)
    );
    
    console.log(`\n✓ Results saved to ${CV_CONFIG.resultsPath}/cv_results.json`);
    
    // 6. Статистическая значимость
    if (aggregated.auc.std < 0.05) {
      console.log('\n✅ Model shows STABLE performance across folds (σ < 0.05)');
    } else {
      console.log('\n⚠️  Model shows VARIABLE performance across folds (σ ≥ 0.05)');
    }
    
    console.log('\n' + '='.repeat(70));
    console.log('✅ Cross-validation completed successfully!');
    console.log('='.repeat(70) + '\n');
    
    // Очистка
    xs.dispose();
    ys.dispose();
    
    return report;
    
  } catch (error) {
    console.error('\n❌ Cross-validation failed:');
    console.error(error);
    process.exit(1);
  }
}

// Запуск при вызове напрямую
if (require.main === module) {
  runCrossValidation();
}

module.exports = { runCrossValidation, createStratifiedFolds };
