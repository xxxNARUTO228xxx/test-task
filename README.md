# ML Regression Pipeline (CatBoost + FastAPI)

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Обучение

```bash
python -m src.train.train --data /path/to/data.csv --config configs/config.yaml
```
- Артефакты сохраняются в `artifacts/`: `model.cbm`, `selected_features.json`, `dropped_features.json`.
- Логи MLflow в `mlruns/`. Запуск UI:

```bash
mlflow ui --backend-store-uri mlruns
```

## Запуск API локально

```bash
export ARTIFACTS_DIR=$(pwd)/artifacts
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Проверка:
```bash
curl -X POST "http://0.0.0.0:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"items":[{"features": {"feature_a": 1.0, "feature_b": "x"}}]}'
```

## Docker

Сборка:
```bash
docker build -t model-api .
```

Запуск:
```bash
docker run --rm -e ARTIFACTS_DIR=/app/artifacts -p 8000:8000 model-api
```

## Тесты

```bash
pytest -q
```
