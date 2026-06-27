<a id="ru"></a>

# Нейромодератор v0.1

**Русский** | [English](#en)

Нейромодератор - это исследовательский ML-проект для автоматической модерации публичных чатов и комментариев. Модель анализирует пользовательский текст и присваивает ему одну или несколько меток токсичности.

Проект сделан на классических методах машинного обучения: TF-IDF-признаки, SVD, мета-признаки текста и бинарные классификаторы для каждой метки. Такой подход выбран как более интерпретируемая и воспроизводимая альтернатива готовым black-box toxic-классификаторам.

### Ноутбуки

- [Русская версия ноутбука](notebooks/NeuroModerator_v_0_1_RU.ipynb)
- [English notebook version](notebooks/NeuroModerator_v_0_1_EN.ipynb)

## Что умеет проект

Модель решает задачу **multi-label classification** и распознаёт 4 типа сообщений:

| Метка | Значение |
|---|---|
| `INSULT` | Оскорбления, агрессивная или унизительная речь |
| `THREAT` | Угрозы и потенциально опасные высказывания |
| `OBSCENITY` | Нецензурная или неприемлемая лексика |
| `NORMAL` | Сообщения без нарушений |

Один комментарий может получить несколько меток одновременно. Например, сообщение может быть одновременно оскорблением и угрозой.

## Особенности проекта

- Поддержка мультиметочной классификации.
- Обучение на русскоязычных и переведённых англоязычных датасетах.
- Использование словаря русской обсценной лексики как дополнительного источника данных.
- Балансировка классов перед обучением.
- Комбинация word-level и char-level TF-IDF-признаков.
- Дополнительные признаки длины текста, количества слов, символов и других текстовых характеристик.
- Сравнение нескольких классических ML-алгоритмов.
- Кэширование признаков и обученных моделей.
- Интерактивный режим проверки комментариев без перезапуска пайплайна.

## Использованные данные

В проекте используются следующие источники данных:

1. `alexandersemiletov/toxic-russian-comments` - основной датасет русских токсичных комментариев.
2. `reihanenamdari/youtube-toxicity-data` - дополнительный англоязычный датасет YouTube-комментариев.
3. `jahangirhussen/toxic-comment` - дополнительный англоязычный датасет токсичных комментариев.
4. `ilyaxin/Russian-Swear-Words` - словарь русской обсценной лексики.

Англоязычные данные фильтруются и переводятся на русский язык с помощью NLLB-модели, чтобы расширить обучающую выборку для русскоязычного нейромодератора.

## Модели и признаки

В проекте тестируются и обучаются следующие алгоритмы:

- `LightGBM`
- `XGBoost`
- `LinearSVC`
- `MLPClassifier`

Основные признаки:

- word TF-IDF;
- char TF-IDF;
- SVD-сжатие TF-IDF-признаков;
- простые мета-признаки текста.

Финальная версия выбирает лучшую стратегию для каждой метки отдельно. В последнем запуске лучшей моделью для всех четырёх меток стала `LightGBM`.

## Итоговый датасет

После очистки, объединения и балансировки:

- размер датасета: **247 305 комментариев**;
- пустые тексты: **0**;
- средняя длина комментария: **14.83 слова**;
- большинство комментариев имеют одну метку;
- часть комментариев содержит две и более метки.

## Результаты

Итоговая оценка на test-выборке:

| Метрика | Значение |
|---|---:|
| `F1 micro` | `0.9896` |
| `F1 macro` | `0.9900` |
| `Hamming loss` | `0.0076` |

Поклассовые результаты на test-выборке:

| Метка | F1 test |
|---|---:|
| `INSULT` | `0.9895` |
| `NORMAL` | `0.9838` |
| `OBSCENITY` | `0.9941` |
| `THREAT` | `0.9927` |

> Важно: метрики получены на подготовленном и сбалансированном датасете. Они показывают качество модели в рамках проведённого эксперимента, но не гарантируют такое же качество на любых реальных пользовательских данных.

## Как запустить

### Вариант 1. Google Colab

1. Откройте ноутбук `Нейромодератор_v_0_1.ipynb` или `NeuroModerator_v_0_1_EN.ipynb`.
2. Включите GPU в настройках среды выполнения.
3. Выполните ячейку с созданием `requirements.txt`.
4. Установите зависимости:

```bash
pip install -r requirements.txt
```

5. Последовательно выполните ячейки ноутбука.

### Вариант 2. Локальный запуск

Для локального запуска рекомендуется:

- Python 3.12;
- 32+ GB RAM;
- динамический файл подкачки или swap 32+ GB;
- CUDA-совместимая видеокарта, если планируется ускорять перевод данных.

Установка зависимостей:

```bash
pip install -r requirements.txt
```

## Интерактивный режим

После обучения или загрузки кэшированных артефактов можно запустить интерактивную проверку комментариев. Пользователь вводит текст, а модель выводит:

- предсказанные метки;
- вероятности по каждой метке;
- итоговое решение нейромодератора.

Пример логики вывода:

```text
TEXT:
<user comment>

PREDICTION:
Labels: INSULT, THREAT

PROBABILITIES:
INSULT:    0.9821
NORMAL:    0.0314
OBSCENITY: 0.2450
THREAT:    0.8756
```

## Почему не готовая токсик-модель

Готовые модели вроде BERT/RuBERT/Detoxify полезны, но в этом проекте был выбран классический ML-подход по нескольким причинам:

- нужна мультиметочная классификация, а не только binary toxic / non-toxic;
- реальные комментарии содержат сленг, ошибки, короткие фразы и нестандартную лексику;
- классические модели проще анализировать, кэшировать и сравнивать;
- проект показывает полный ML-пайплайн: сбор данных, очистку, балансировку, признаки, обучение, оценку и интерактивный режим.

## Ограничения

- Модель не является полноценной production-системой модерации.
- Возможны ошибки на сарказме, контексте, завуалированных угрозах и новых формах токсичной речи.
- Метка `INSULT` сейчас объединяет несколько разных типов токсичности.
- Качество зависит от домена, языка и стиля пользовательских сообщений.
- Перед использованием в реальном продукте требуется дополнительная проверка на свежих данных.

## Возможные улучшения

- Добавить больше реальных размеченных комментариев.
- Расширить набор меток: `TOXIC`, `RACISM`, `RELIGION_HATE`, `SEXISM`, `SPAM` и другие.
- Добавить fastText или sentence embeddings.
- Использовать RuBERT/transformer-модель как отдельный baseline или часть ансамбля.
- Сделать API на FastAPI.
- Добавить простой web-интерфейс для демонстрации модели.
- Настроить CI-проверку ноутбука и линтинг кода.

## Статус проекта

`v0.1` - исследовательская версия проекта. Основная цель версии - показать полный воспроизводимый ML-пайплайн для русскоязычной мультиметочной модерации комментариев.

---

<a id="en"></a>

# NeuroModerator v0.1

[Русский](#ru) | **English**

NeuroModerator is a research-oriented machine learning project for automated moderation of public chats and user comments. The model analyzes user text and assigns one or more toxicity labels to it.

The project is built with classical machine learning methods: TF-IDF features, SVD, text-level meta-features, and binary classifiers for each label. This approach was chosen as a more interpretable and reproducible alternative to ready-made black-box toxicity classifiers.

### Notebooks

- [English notebook version](notebooks/NeuroModerator_v_0_1_EN.ipynb)
- [Russian notebook version](notebooks/NeuroModerator_v_0_1_RU.ipynb)

## What the project does

The model solves a **multi-label classification** task and detects 4 types of messages:

| Label | Meaning |
|---|---|
| `INSULT` | Insults, aggressive or degrading language |
| `THREAT` | Threats and potentially dangerous statements |
| `OBSCENITY` | Obscene or inappropriate language |
| `NORMAL` | Messages without violations |

A single comment can receive several labels at the same time. For example, a message can be both an insult and a threat.

## Project features

- Multi-label classification support.
- Training on Russian and translated English-language datasets.
- Use of a Russian obscene vocabulary dictionary as an additional data source.
- Class balancing before model training.
- Combination of word-level and char-level TF-IDF features.
- Additional text-length, word-count, character-count and other simple text meta-features.
- Comparison of several classical ML algorithms.
- Caching of features and trained models.
- Interactive comment-checking mode without restarting the pipeline.

## Data sources

The notebook uses the following data sources:

1. `alexandersemiletov/toxic-russian-comments` — the main Russian toxic comment dataset.
2. `reihanenamdari/youtube-toxicity-data` — an additional English-language YouTube toxicity dataset.
3. `jahangirhussen/toxic-comment` — an additional English-language toxic comment dataset.
4. `ilyaxin/Russian-Swear-Words` — a Russian obscene vocabulary dictionary.

English-language data is filtered and translated into Russian with an NLLB model to expand the training set for the Russian-language moderation model.

## Models and features

The project trains and evaluates the following algorithms:

- `LightGBM`
- `XGBoost`
- `LinearSVC`
- `MLPClassifier`

Main features:

- word TF-IDF;
- char TF-IDF;
- SVD-compressed TF-IDF features;
- simple text meta-features.

The final version selects the best prediction strategy for each label separately. In the latest run, `LightGBM` was selected as the best model for all four labels.

## Final dataset

After cleaning, merging and balancing:

- dataset size: **247,305 comments**;
- empty texts: **0**;
- average comment length: **14.83 words**;
- most comments have one label;
- some comments contain two or more labels.

## Results

Final evaluation on the test split:

| Metric | Value |
|---|---:|
| `F1 micro` | `0.9896` |
| `F1 macro` | `0.9900` |
| `Hamming loss` | `0.0076` |

Per-label results on the test split:

| Label | F1 test |
|---|---:|
| `INSULT` | `0.9895` |
| `NORMAL` | `0.9838` |
| `OBSCENITY` | `0.9941` |
| `THREAT` | `0.9927` |

> Important: these metrics were obtained on a prepared and balanced dataset. They describe the quality of the performed experiment, but they do not guarantee the same performance on arbitrary real-world user data.

## Repository structure

Recommended repository structure:

```text
.
├── README.md
├── Нейромодератор_v_0_1.ipynb
├── NeuroModerator_v_0_1_EN.ipynb
├── requirements.txt
├── models_cache/              # trained model cache, if published separately
├── feature_cache/             # TF-IDF, SVD and meta-feature cache, if published separately
└── artifacts/                 # additional experiment artifacts
```

## How to run

### Option 1. Google Colab

1. Open `Нейромодератор_v_0_1.ipynb` or `NeuroModerator_v_0_1_EN.ipynb`.
2. Enable GPU in the runtime settings.
3. Run the cell that creates `requirements.txt`.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the notebook cells sequentially.

### Option 2. Local run

For a local run, the recommended setup is:

- Python 3.12;
- 32+ GB RAM;
- dynamic page file or 32+ GB swap;
- CUDA-compatible GPU if you plan to accelerate data translation.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Interactive mode

After training or loading cached artifacts, the notebook can run an interactive comment-checking mode. The user enters a text, and the model outputs:

- predicted labels;
- probabilities for each label;
- the final moderation decision.

Example output logic:

```text
TEXT:
<user comment>

PREDICTION:
Labels: INSULT, THREAT

PROBABILITIES:
INSULT:    0.9821
NORMAL:    0.0314
OBSCENITY: 0.2450
THREAT:    0.8756
```

## Why not use a ready-made toxicity model

Ready-made models such as BERT/RuBERT/Detoxify are useful, but this project uses a classical ML approach for several reasons:

- the task requires multi-label classification, not just binary toxic / non-toxic prediction;
- real comments contain slang, typos, short phrases and non-standard vocabulary;
- classical models are easier to analyze, cache and compare;
- the project demonstrates a complete ML pipeline: data collection, cleaning, balancing, feature engineering, training, evaluation and interactive inference.

## Limitations

- The model is not a full production moderation system.
- It can make mistakes on sarcasm, context-dependent toxicity, hidden threats and new forms of abusive language.
- The `INSULT` label currently combines several different toxicity patterns.
- Quality depends on the domain, language and style of user messages.
- Additional validation on fresh data is required before real-world use.

## Future improvements

- Add more real labeled comments.
- Extend the label set: `TOXIC`, `RACISM`, `RELIGION_HATE`, `SEXISM`, `SPAM` and others.
- Add fastText or sentence embeddings.
- Use RuBERT or another transformer model as a separate baseline or as part of an ensemble.
- Build a FastAPI API.
- Add a simple web interface for model demonstration.
- Configure CI notebook checks and code linting.

## Project status

`v0.1` is a research version of the project. The main goal of this version is to demonstrate a complete reproducible ML pipeline for Russian-language multi-label comment moderation.
