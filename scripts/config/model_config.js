// scripts/config/model_config.js
module.exports = {
  // Путь к данным (может быть изменен через аргументы командной строки)
  DATA_PATH: './data/gastrectomy_patients.json',
  MODEL_SAVE_PATH: './models/gastrectomy_model',
  SCALER_SAVE_PATH: './models/scaler.json',
  METRICS_SAVE_PATH: './reports/metrics.json',
  
  // Параметры модели
  MODEL_ARCHITECTURE: {
    inputSize: 13, // age, bmi, op_time, blood_loss, stage_IIB, stage_IIIA, ... (one-hot)
    layers: [
      { units: 64, activation: 'relu', dropout: 0.3 },
      { units: 32, activation: 'relu', dropout: 0.2 },
      { units: 1, activation: 'sigmoid' } // Для бинарной классификации осложнений
    ]
  },
  
  // Параметры обучения
  TRAINING: {
    epochs: 100,
    batchSize: 32,
    validationSplit: 0.2,
    optimizer: 'adam',
    loss: 'binaryCrossentropy'
  },
  
  // Целевая переменная
  TARGET_COLUMN: 'complications',
  
  // Признаки (после предобработки)
  FEATURE_COLUMNS: [
    'age', 'bmi', 'operation_time_min', 'blood_loss_ml', 'lymph_nodes_removed',
    'sex_M', 'tumor_stage_IIB', 'tumor_stage_IIIA', 'tumor_stage_IIA', 'tumor_stage_IIIB',
    'surgery_type_laparoscopic', 'neoadjuvant_therapy_true'
  ]
};
