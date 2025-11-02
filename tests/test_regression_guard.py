import json
import os
import tempfile

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from fastapi.testclient import TestClient

from src.api.main import app


def _train_baseline(tmpdir: str, model_name: str = "catboost-regressor-20251101") -> tuple:
    """Обучает базовую модель и возвращает предсказания для тестовых примеров.
    
    Args:
        tmpdir: Временная директория для сохранения артефактов
        model_name: Имя модели
        
    Returns:
        tuple: (endpoint, predictions_dict) где predictions_dict содержит предсказания для каждого примера
    """
    # Простая линейная зависимость для воспроизводимости
    rng = np.random.default_rng(0)
    x = np.arange(50)
    y = x.astype(float) + rng.normal(0, 0.1, size=50)
    df = pd.DataFrame({"x": x, "y": y})
    m = CatBoostRegressor(loss_function="RMSE", depth=4, learning_rate=0.1, random_seed=0, verbose=False)
    m.fit(df[["x"]], df["y"])

    os.makedirs(tmpdir, exist_ok=True)
    m.save_model(os.path.join(tmpdir, "model.cbm"))
    with open(os.path.join(tmpdir, "selected_features.json"), "w", encoding="utf-8") as f:
        f.write('{"numerical_features":["x"],"categorical_features":[]}')

    # Создаем тестовые примеры (10 примеров)
    test_examples = [{"x": float(i)} for i in range(10)]
    test_predictions = m.predict(pd.DataFrame(test_examples))
    
    # Сохраняем информацию о модели для использования в API
    model_info = {
        "model_name": model_name,
        "endpoint": f"/{model_name}",
        "sample_indices": list(range(10)),
        "targets": ["y"],
        "is_multi_target": False,
        "test_examples": {
            "features": test_examples,
            "targets_true": {
                "y": test_predictions.tolist()  # Используем предсказания как "истинные" значения для теста
            }
        }
    }
    
    with open(os.path.join(tmpdir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    endpoint = f"/{model_name}"
    predictions_dict = {i: float(pred) for i, pred in enumerate(test_predictions)}
    
    return endpoint, predictions_dict


def test_regression_prediction_tolerance() -> None:
    """Тест на устойчивость предсказаний модели с использованием сохраненных примеров."""
    model_name = "catboost-regressor-20251101"
    function_name = f"predict_{model_name.replace('-', '_')}"
    
    with tempfile.TemporaryDirectory() as d:
        endpoint, baseline_predictions = _train_baseline(d, model_name)
        os.environ["ARTIFACTS_DIR"] = d
        
        # Создаем клиент для тестирования
        # TestClient автоматически вызывает startup события при первом запросе
        client = TestClient(app)
        
        # Загружаем тестовые примеры из model_info.json
        with open(os.path.join(d, "model_info.json"), "r", encoding="utf-8") as f:
            model_info = json.load(f)
        
        test_examples = model_info["test_examples"]["features"]
        
        # Тестируем каждый пример
        for idx, example_features in enumerate(test_examples):
            resp = client.post(endpoint, json={"items": [{"features": example_features}]})
            assert resp.status_code == 200
            pred = float(resp.json()["predictions"][0])
            baseline = baseline_predictions[idx]
            # Допуск по отклонению
            assert abs(pred - baseline) < 0.5, f"Пример {idx}: pred={pred}, baseline={baseline}"
