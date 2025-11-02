import json
import os
import tempfile

import pandas as pd
from catboost import CatBoostRegressor
from fastapi.testclient import TestClient

from src.api.main import app


def test_catboost_regressor_20251101() -> None:
    """Тест успешного предсказания через эндпойнт."""
    model_name = "catboost-regressor-20251101"
    function_name = f"{model_name.replace('-', '_')}"
    
    with tempfile.TemporaryDirectory() as d:
        endpoint = f"/{model_name}"
        os.environ["ARTIFACTS_DIR"] = d
        
        # Создаем клиент для тестирования
        # FastAPI автоматически регистрирует эндпойнт при startup
        client = TestClient(app)
        payload = {"items": [{"features": {"x": 5}}]}
        resp = client.post(endpoint, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 1
