// scripts/utils/data_loader.js
const fs = require('fs');

function loadData(filePath) {
  return new Promise((resolve, reject) => {
    fs.readFile(filePath, 'utf8', (err, data) => {
      if (err) {
        console.error('Error reading data file:', err);
        reject(err);
        return;
      }
      try {
        const jsonData = JSON.parse(data);
        console.log(`Data loaded successfully. Shape: [${jsonData.length}, ?]`);
        resolve(jsonData);
      } catch (parseErr) {
        console.error('Error parsing JSON:', parseErr);
        reject(parseErr);
      }
    });
  });
}

module.exports = { loadData };
