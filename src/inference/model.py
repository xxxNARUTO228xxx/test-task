from typing import Dict, List, Optional, Union
import json
import os
import pandas as pd
from catboost import CatBoostRegressor


class ModelInfo:
    """Информация о загруженной модели."""

    def __init__(
        self,
        model: CatBoostRegressor,
        feature_order: List[str],
        numerical_features: List[str],
        categorical_features: List[str],
    ) -> None:
        self.model = model
        self.feature_order = feature_order
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features


class ModelService:
    """Сервис инференса: загрузка нескольких моделей и предсказание.

    Может загружать и хранить несколько моделей, каждая со своим именем.
    Ожидает для каждой модели:
    - artifacts/{model_name}/model.cbm
    - artifacts/{model_name}/selected_features.json {"numerical_features": [...], "categorical_features": [...]}
    """

    def __init__(self) -> None:
        self._models: Dict[str, ModelInfo] = {}

    def load_models(self, artifacts_dir: str, model_names: List[str]) -> None:
        """Загружает все модели из списка в указанной директории.

        Args:
            artifacts_dir: Базовая директория с артефактами
            model_names: Список имен моделей для загрузки
        """
        for model_name in model_names:
            model_dir = os.path.join(artifacts_dir, model_name)
            if os.path.exists(model_dir):
                try:
                    model, feature_order, numerical_features, categorical_features = self._load_single_model(model_dir)
                    self._models[model_name] = ModelInfo(
                        model, feature_order, numerical_features, categorical_features
                    )
                except Exception as e:
                    print(f"Ошибка при загрузке модели {model_name}: {e}")
            else:
                print(f"Предупреждение: директория модели {model_name} не найдена: {model_dir}")

    def _load_single_model(self, model_dir: str) -> tuple[CatBoostRegressor, List[str], List[str], List[str]]:
        """Загружает одну модель из указанной директории.

        Args:
            model_dir: Директория с артефактами модели

        Returns:
            tuple: (модель CatBoostRegressor, список фичей в порядке, числовые фичи, категориальные фичи)
        """
        model_path = os.path.join(model_dir, "model.cbm")
        sel_path = os.path.join(model_dir, "selected_features.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Не найден файл модели: {model_path}")
        if not os.path.exists(sel_path):
            raise FileNotFoundError(f"Не найден файл признаков: {sel_path}")

        with open(sel_path, "r", encoding="utf-8") as f:
            selected = json.load(f)
        numerical_features = selected.get("numerical_features", [])
        categorical_features = selected.get("categorical_features", [])
        feature_order = numerical_features + categorical_features

        model = CatBoostRegressor()
        model.load_model(model_path)
        return model, feature_order, numerical_features, categorical_features

    def predict(self, model_name: str, items: List[Dict[str, Optional[Union[float, int, str]]]]) -> List[float]:
        """Возвращает список предсказаний для входных объектов.

        Args:
            model_name: Имя модели для использования
            items: Список словарей с фичами для предсказания

        Returns:
            List[float]: Список предсказаний. Для мультитаргетной модели каждое предсказание
                        может быть списком значений, который затем разворачивается
        
        Порядок признаков фиксирован для каждой модели. Отсутствующие значения → None.
        """
        if model_name not in self._models:
            raise RuntimeError(f"Модель {model_name} не загружена")
        
        model_info = self._models[model_name]
        rows: List[List[Optional[Union[float, int, str]]]] = []
        for obj in items:
            row = [obj.get(name, None) for name in model_info.feature_order]
            rows.append(row)
        
        # Создаем DataFrame
        X = pd.DataFrame(rows, columns=model_info.feature_order)
        
        # Обрабатываем None значения: для категориальных фичей преобразуем None в строку "None"
        # Это соответствует обработке при обучении (build_pools использует fillna("None"))
        for cat_col in model_info.categorical_features:
            if cat_col in X.columns:
                X[cat_col] = X[cat_col].fillna("None")
        
        preds = model_info.model.predict(X)
        
        # Если модель мультитаргетная, preds будет массивом с несколькими колонками
        # Форма: (n_samples, n_targets) для мультитаргета или (n_samples,) для одного таргета
        if len(preds.shape) > 1 and preds.shape[1] > 1:
            # Мультитаргетная модель: возвращаем все предсказания для всех объектов как плоский список
            return [float(p) for pred in preds for p in pred]
        else:
            # Один таргет: возвращаем список предсказаний
            return [float(p) for p in preds]
