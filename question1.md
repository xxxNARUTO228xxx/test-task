# Вопрос 1
Представьте, что модель обучена, протестирована локально и показывает хорошее качество.

Опишите, как вы выведете её в production, если стек: FastAPI + Docker + MLflow + ClickHouse + GitLab CI/CD.В ответе укажите:

- структуру репозитория,
- шаги пайплайна CI/CD,
- где будет храниться модель и как подгружаться в сервис инференса,
- что и где будете мониторить после релиза.

# Ответ
## Структура репозитория:
```
project/
├── app.py                  # основной файл приложения FastAPI
├── config.py               # конфиг проекта
├── models/                 # папка с файлами моделей
├── requirements.txt        # зависимости Python
├── Dockerfile              # инструкции сборки Docker-образа
├── .gitlab-ci.yml          # конфиг пайплайна CI/CD
├── mlflow/                 # директория MlFlow
│   ├── experiments.csv     # метаданные экспериментов
│   ├── runs/               # записи результатов запусков моделей
│   └── artifacts/          # артефакты mlflow
├── db/                     # скрипты инициализации базы данных
│   └── schema.sql          # схема ClickHouse
└── tests/                  # тесты
    ├── test_app.py         # тестирование основного API-приложения, io-тесты, парсинг, сборка фичей
    └── test_model.py       # тестирование эндпойнтов и наличия регрессии
```

## Шаги пайплайна CI/CD
1. Проверка изменений (lint):

    Выполняются статический анализ и проверка стиля кода перед сборкой образа.

    ```
    stages:
    - lint
    - build
    - deploy
    - monitor
    test-lint:
    stage: lint
    script:
        - pip install flake8 black isort mypy pylint
        - flake8 .
        - black --check .
        - isort --check-only .
        - mypy project
        - pylint project
    ```

2. Сборка Docker-образа (build):

    Образ собирается автоматически каждый раз при пушинге новых коммитов в ветку main.

    ```
    docker-build:
    stage: build
    image: docker:latest
    services:
        - docker:dind
    script:
        - docker login -u gitlab-ci-token -p $CI_JOB_TOKEN registry.gitlab.com/project-name
        - docker build -t registry.gitlab.com/project-name/app:$CI_COMMIT_SHA .
        - docker push registry.gitlab.com/project-name/app:$CI_COMMIT_SHA
    ```
3. Развертывание в production (deploy):

    Производится деплой контейнера в production-среде, подключается база данных ClickHouse и загружаются необходимые зависимости.

    ```
    production-deploy:
    stage: deploy
    environment: production
    only:
        - main
    when: manual
    script:
        - docker pull registry.gitlab.com/project-name/app:$CI_COMMIT_SHA
        - docker run -d --name app-production -p 8080:8080 \
        -v /path/to/db:/var/lib/clickhouse \
        -e DATABASE_URL=clickhouse://host:port/database_name \
        registry.gitlab.com/project-name/app:$CI_COMMIT_SHA
    ```
## Хранение и загрузка модели
Модель подгружается из S3\minio при старте службы FastAPI в каталог models/

## Метрики и мониторинг сервисов
Для мониторинга системы рекомендуется использовать Prometheus+Grafana, интегрированные с системой логирования (ELK Stack).

Что будем мониторить:
- HTTP-метрики: количество запросов, среднее время отклика, процент ошибок.
- Загрузка CPU, памяти и диска контейнера Docker.
- Метрики БД: размер таблиц, объем хранимых данных, производительность запросов.
- Время предикта: измеряется время обработки каждого запроса модели.
- Динамику изменения предсказаний модели и фичей.
    - Каждый новый запрос добавляет новое значение предсказанной величины.
    - Прометей собирает эти значения и строит временные ряды.
    - Grafana визуально отображает динамику средней или медианной оценки предсказаний.

Настройка Grafana Dashboards:
- Панель мониторинга ресурсов сервера.
- Панель производительности модели (скорость обработки).
- Общие показатели здоровья инфраструктуры (CPU, RAM, сетевые запросы).
- Предсказания модели и признаки в динамике