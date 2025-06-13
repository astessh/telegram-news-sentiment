# News Sentiment Classification

Классификация новостных заголовков по тональности: положительная (1), нейтральная (0.5), отрицательная (0). Цель проекта — создать модель, способную автоматически классифицировать краткие тексты новостей по степени эмоциональной окрашенности.

## Задача

Медиа ежедневно публикуют множество коротких новостных заметок, часто с субъективной подачей. Автоматическая оценка тональности таких новостей может помочь:

* отслеживать эмоциональный фон новостного фона;
* фильтровать токсичные или провокационные заголовки;
* создавать рекомендательные системы на основе пользовательских предпочтений.

Мы решаем задачу бинарной классификации:

* `1` — положительные или нейтральные новости
* `0` — негативные новости

(нейтральные (`0.5`) учитываются только на этапе аннотации и убираются в baseline'е)


## Setup

Проект использует [Poetry](https://python-poetry.org/) для управления зависимостями и [Hydra](https://hydra.cc/) для конфигурации. Все основные скрипты находятся в директории `telegram_news_sentiment/`.

### Установка

```bash
git clone git@github.com:astessh/telegram-news-sentiment.git
cd telegram-news-sentiment

poetry install

poetry env activate 

poetry run dvc init 
```


## Train

В проекте реализованы две модели:

1. **Baseline-модель (TF-IDF + Logistic Regression)**
2. **Основная модель: Qwen 0.5B + P-Tuning v2**

### Подготовка данных

Сырые данные нужно положить в `data/raw/data.json`, формат:

```json
{"text": "Заголовок новости", "target": 1.0}
```

можно скачать с помощью
```bash
poetry run python3 telegram_news_sentiment/data_loading.py                            
```

Запустить препроцессинг (чистка, токенизация, сплит):

```bash
poetry run python telegram_news_sentiment/preprocessing.py
```

Созданные файлы появятся в `data/processed/`:

* `train.json`
* `val.json`
* `test.json`


### Запуск обучения baseline-модели

```bash
PYTHONPATH=./telegram_news_sentiment poetry run python telegram_news_sentiment/baseline/train.py
```
Настройка в `configs/baseline.yaml`.
Результаты сохраняются в `telegram_news_sentiment/baseline/outputs/`.


### Запуск обучения Qwen 0.5B

```bash
PYTHONPATH=./telegram_news_sentiment poetry run python telegram_news_sentiment/qwen/train_qwen.py
```

Модель использует: `Qwen/Qwen-0.5B` с адаптацией через Ptune

## Структура проекта

```
.
├── configs/               # hydra-конфиги
├── data/                 
│   ├── raw/              # исходные данные
│   └── processed/        # preprocessed и сплиты
├── telegram_news_sentiment/
│   ├── baseline/         # TF-IDF baseline
│   ├── models/           # LLM и p-tuning
│   ├── outputs/          # сохраненные модели
│   └── qwen/             # сохраненные модели
```

