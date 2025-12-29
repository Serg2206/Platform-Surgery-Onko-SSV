// scripts/validation/data_validator.js
const REQUIRED_COLUMNS = [
  'age', 'sex', 'bmi', 'tumor_stage', 'surgery_type', 'operation_time_min',
  'blood_loss_ml', 'complications', 'neoadjuvant_therapy', 'lymph_nodes_removed'
];

function validateDataStructure(data) {
  if (!Array.isArray(data) || data.length === 0) {
    throw new Error('Data must be a non-empty array.');
  }

  const firstRow = data[0];
  for (const col of REQUIRED_COLUMNS) {
    if (!(col in firstRow)) {
      throw new Error(`Missing required column: ${col}`);
    }
    // Проверка типа (можно расширить)
    if (col === 'age' && typeof firstRow[col] !== 'number') {
       throw new Error(`Column ${col} must be a number, got ${typeof firstRow[col]} for row 0.`);
    }
  }

  console.log('Data structure validation passed.');
}

module.exports = { validateDataStructure, REQUIRED_COLUMNS };
