from typing import Dict, List, Any
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Считает RMSE, MAE, R2."""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def ensure_dir(path: str) -> None:
    """Создаёт директорию, если её нет."""
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    """Сохраняет объект в JSON с utf-8 и отступами."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """Загружает JSON-файл и возвращает объект."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
