/**
 * Gastrectomy Risk Model Training Script
 * Обучение модели для предсказания рисков после гастрэктомии
 * 
 * Features:
 * - Полный ML pipeline с TensorFlow.js
 * - Предобработка клинических данных
 * - Обучение нейронной сети
 * - Сохранение обученной модели
 * - Метрики качества (AUC, Accuracy, Precision, Recall)
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs').promises;
const path = require('path');

// Конфигурация модели
const CONFIG = {
  modelPath: './models/gastrectomy_risk',
  dataPath: './data/gastrectomy_patients.json',
  epochs: 100,
  batchSize: 32,
  validationSplit: 0.2,
  learningRate: 0.001,
  features: [
    'age',
    'bmi',
    'albumin',
    'hemoglobin', 
    'asa_score',
    'tumor_stage',
    'comorbidity_count',
    'previous_surgery'
  ],
  target: 'major_complication'
};

/**
 * Загрузка и предобработка данных
 */
async function loadAndPreprocessData() {
  console.log('📊 Loading training data...');
  
  const rawData = JSON.parse(
    await fs.readFile(CONFIG.dataPath, 'utf-8')
  );
  
  console.log(`✓ Loaded ${rawData.length} patient records`);
  
  // Нормализация признаков
  const featureData = rawData.map(patient => 
    CONFIG.features.map(feature => patient[feature] || 0)
  );
  
  const labels = rawData.map(patient => 
    patient[CONFIG.target] ? 1 : 0
  );
  
  // Создание тензоров
  const xs = tf.tensor2d(featureData);
  const ys = tf.tensor2d(labels, [labels.length, 1]);
  
  // Z-score нормализация
  const mean = xs.mean(0);
  const std = xs.sub(mean).square().mean(0).sqrt();
  const xsNormalized = xs.sub(mean).div(std.add(1e-7));
  
  console.log('✓ Data preprocessing complete');
  console.log(`  Features: ${CONFIG.features.length}`);
  console.log(`  Samples: ${rawData.length}`);
  console.log(`  Positive cases: ${labels.filter(l => l === 1).length}`);
  
  return { xs: xsNormalized, ys, mean, std };
}

/**
 * Создание архитектуры нейронной сети
 */
function createModel(inputDim) {
  console.log('\n🧠 Building neural network...');
  
  const model = tf.sequential({
    layers: [
      // Входной слой
      tf.layers.dense({
        inputShape: [inputDim],
        units: 64,
        activation: 'relu',
        kernelInitializer: 'heNormal'
      }),
      
      // Dropout для регуляризации
      tf.layers.dropout({ rate: 0.3 }),
      
      // Скрытый слой
      tf.layers.dense({
        units: 32,
        activation: 'relu',
        kernelInitializer: 'heNormal'
      }),
      
      tf.layers.dropout({ rate: 0.2 }),
      
      // Выходной слой (бинарная классификация)
      tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
      })
    ]
  });
  
  // Компиляция модели
  model.compile({
    optimizer: tf.train.adam(CONFIG.learningRate),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy', 'precision', 'recall']
  });
  
  console.log('✓ Model architecture:');
  model.summary();
  
  return model;
}

/**
 * Обучение модели с callbacks
 */
async function trainModel(model, xs, ys) {
  console.log('\n🎯 Training model...');
  console.log(`  Epochs: ${CONFIG.epochs}`);
  console.log(`  Batch size: ${CONFIG.batchSize}`);
  console.log(`  Validation split: ${CONFIG.validationSplit}`);
  
  const history = await model.fit(xs, ys, {
    epochs: CONFIG.epochs,
    batchSize: CONFIG.batchSize,
    validationSplit: CONFIG.validationSplit,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 10 === 0) {
          console.log(
            `  Epoch ${epoch + 1}/${CONFIG.epochs} - ` +
            `loss: ${logs.loss.toFixed(4)}, ` +
            `acc: ${logs.acc.toFixed(4)}, ` +
            `val_loss: ${logs.val_loss.toFixed(4)}, ` +
            `val_acc: ${logs.val_acc.toFixed(4)}`
          );
        }
      }
    }
  });
  
  console.log('✓ Training complete!');
  
  return history;
}

/**
 * Вычисление AUC-ROC
 */
function calculateAUC(predictions, labels) {
  const pairs = predictions.map((pred, i) => ({
    pred: pred[0],
    label: labels[i]
  }));
  
  pairs.sort((a, b) => b.pred - a.pred);
  
  let positives = 0;
  let negatives = 0;
  let sumRanks = 0;
  
  pairs.forEach((pair, i) => {
    if (pair.label === 1) {
      positives++;
      sumRanks += i + 1;
    } else {
      negatives++;
    }
  });
  
  const auc = (sumRanks - (positives * (positives + 1)) / 2) / 
              (positives * negatives);
  
  return auc;
}

/**
 * Оценка модели на валидационных данных
 */
async function evaluateModel(model, xs, ys) {
  console.log('\n📈 Evaluating model...');
  
  const predictions = await model.predict(xs).array();
  const labels = await ys.array();
  
  // Вычисление метрик
  const auc = calculateAUC(predictions, labels.flat());
  
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
  
  const metrics = {
    auc,
    accuracy,
    precision,
    recall,
    f1,
    tp, fp, tn, fn
  };
  
  console.log('\n✓ Model Performance:');
  console.log(`  AUC-ROC: ${auc.toFixed(4)}`);
  console.log(`  Accuracy: ${accuracy.toFixed(4)}`);
  console.log(`  Precision: ${precision.toFixed(4)}`);
  console.log(`  Recall: ${recall.toFixed(4)}`);
  console.log(`  F1-Score: ${f1.toFixed(4)}`);
  console.log(`\n  Confusion Matrix:`);
  console.log(`    TP: ${tp}, FP: ${fp}`);
  console.log(`    FN: ${fn}, TN: ${tn}`);
  
  return metrics;
}

/**
 * Сохранение модели и метаданных
 */
async function saveModel(model, metrics, mean, std) {
  console.log('\n💾 Saving model...');
  
  await model.save(`file://${CONFIG.modelPath}`);
  
  // Сохранение метаданных
  const metadata = {
    version: '1.0.0',
    trainedAt: new Date().toISOString(),
    config: CONFIG,
    metrics,
    normalization: {
      mean: await mean.array(),
      std: await std.array()
    }
  };
  
  await fs.writeFile(
    path.join(CONFIG.modelPath, 'metadata.json'),
    JSON.stringify(metadata, null, 2)
  );
  
  console.log(`✓ Model saved to ${CONFIG.modelPath}`);
  console.log('✓ Metadata saved');
}

/**
 * Главная функция обучения
 */
async function main() {
  console.log('\n' + '='.repeat(60));
  console.log('🏥 GASTRECTOMY RISK MODEL TRAINING');
  console.log('='.repeat(60) + '\n');
  
  try {
    // 1. Загрузка данных
    const { xs, ys, mean, std } = await loadAndPreprocessData();
    
    // 2. Создание модели
    const model = createModel(CONFIG.features.length);
    
    // 3. Обучение
    await trainModel(model, xs, ys);
    
    // 4. Оценка
    const metrics = await evaluateModel(model, xs, ys);
    
    // 5. Сохранение
    await saveModel(model, metrics, mean, std);
    
    console.log('\n' + '='.repeat(60));
    console.log('✅ Training pipeline completed successfully!');
    console.log('='.repeat(60) + '\n');
    
    // Очистка памяти
    xs.dispose();
    ys.dispose();
    mean.dispose();
    std.dispose();
    
  } catch (error) {
    console.error('\n❌ Training failed:');
    console.error(error);
    process.exit(1);
  }
}

// Запуск при вызове напрямую
if (require.main === module) {
  main();
}

module.exports = { main, loadAndPreprocessData, createModel, trainModel };
