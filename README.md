# Surgery-Onko-SSV

Платформенный многомодульный научно-практический оригинальный репозиторий SSV.

## О проекте

Создан в качестве платформы для работы с молекулярными данными и анализа их в контексте хирургии и онкологии. Проект включает в себя следующие функции:

* Анализ молекулярных данных
* Предобработка и анализ данных
* Работа с обученными моделями

## Технологии

* JavaScript (Node.js)
* TensorFlow.js
* pandas-js
* Express.js

## Установка

1. Клонировать репозиторий: `git clone https://github.com/Serg2206/Platform-Surgery-Onko-SSV.git`
2. Установить dependencies: `npm install`
3. Восполнение данных из базы данных или другого источника
4. Начало работы с проектом

## Документация

Документация проекта находится в файле `doc/README.md`.

## Тестирование

Тестовые данные и файлы с данными можно найти в `data/`.

## Структура проекта

```
Platform-Surgery-Onko-SSV/
├── .github/
│   └── workflows/        # CI/CD workflows
├── data/
│   └── gastrectomy_patients.json  # Sample patient data
├── doc/
│   └── research_methodology_guide.md  # Research methodology guide
├── scripts/
│   ├── train_gastrectomy_model.js      # ML model training
│   └── cross_validate_gastrectomy.js   # K-fold cross-validation
├── src/
│   └── index.js          # Express API server
├── tests/
│   └── app.test.js       # Jest unit tests
├── package.json          # Dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Использование ML моделей

### Обучение модели гастрэктомии

```bash
node scripts/train_gastrectomy_model.js
```

Модель использует:
- TensorFlow.js для нейронных сетей
- danfojs для обработки данных
- Функции предсказания выживаемости пациентов

### Кросс-валидация

```bash
node scripts/cross_validate_gastrectomy.js
```

Выполняет 5-fold cross-validation для оценки качества модели.

## API Endpoints

### GET /health
Проверка статуса сервера

### POST /predict
Предсказание результатов хирургического вмешательства

**Request body:**
```json
{
  "age": 62,
  "sex": "M",
  "bmi": 24.5,
  "tumor_stage": "IIB",
  "surgery_type": "laparoscopic"
}
```

## Тестирование

```bash
npm test
```

Запускает Jest тесты для проверки функциональности.

## Научная документация

Полное руководство по методологии научных исследований доступно в `doc/research_methodology_guide.md`.

## Лицензия

MIT License
