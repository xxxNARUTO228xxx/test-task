import os
from typing import List
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI

from src.data.schema import PredictRequest, PredictResponse
from src.inference.model import ModelService


# Список моделей для загрузки при старте сервиса
MODELS: List[str] = [
    "catboost-regressor-20251102",
]


app = FastAPI(title="Model API", version="0.1.0")
_service = ModelService()


@app.on_event("startup")
def _startup() -> None:
    """Загружает все модели из списка при старте сервера."""
    artifacts_dir = os.getenv("ARTIFACTS_DIR", os.path.abspath("artifacts"))
    _service.load_models(artifacts_dir, MODELS)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/catboost-regressor-20251102", response_model=PredictResponse)
def catboost_regressor_20251102(req: PredictRequest) -> PredictResponse:
    """Эндпойнт для предсказания модели catboost-regressor-20251102."""
    request_id = uuid4()
    timestamp = datetime.now().isoformat()
    preds = _service.predict("catboost-regressor-20251102", [req.features])
    return PredictResponse(
        request_id=request_id,
        timestamp=timestamp,
        predictions=preds
    )
