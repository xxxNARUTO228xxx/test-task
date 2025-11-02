from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd


def compute_missing_ratio(series: pd.Series) -> float:
    """Возвращает долю пропусков в серии."""
    total = len(series)
    if total == 0:
        return 0.0
    return float(series.isna().sum()) / float(total)


def select_features(
    df_train: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    missing_threshold: float = 0.5,
    corr_threshold: float = 0.95,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """Отбирает фичи по правилам:
    1) Удаляет фичи с высокой долей пропусков (reason: low_hit)
    2) Удаляет константные фичи (reason: constant)
    3) Удаляет сильно коррелирующие числовые фичи (reason: correlation)

    Возвращает:
    - selected: {"numerical_features": [...], "categorical_features": [...]} в фиксированном порядке
    - dropped: {"numerical_features": {name: reason}, "categorical_features": {name: reason}}
    """
    dropped: Dict[str, Dict[str, str]] = {
        "numerical_features": {},
        "categorical_features": {},
    }

    # 1) Фильтр пропусков
    def filter_missing(names: List[str], kind: str) -> List[str]:
        kept: List[str] = []
        for col in names:
            miss = compute_missing_ratio(df_train[col])
            if miss > missing_threshold:
                dropped[kind][col] = "low_hit"
            else:
                kept.append(col)
        return kept

    numerical_kept = filter_missing(numerical_features, "numerical_features")
    categorical_kept = filter_missing(categorical_features, "categorical_features")

    # 2) Фильтр константности
    def filter_constant(names: List[str], kind: str) -> List[str]:
        kept: List[str] = []
        for col in names:
            uniques = pd.unique(df_train[col].dropna())
            if len(uniques) <= 1:
                dropped[kind][col] = "constant"
            else:
                kept.append(col)
        return kept

    numerical_kept = filter_constant(numerical_kept, "numerical_features")
    categorical_kept = filter_constant(categorical_kept, "categorical_features")

    # 3) Фильтр корреляций (только числовые)
    def filter_correlation(names: List[str]) -> List[str]:
        if len(names) <= 1:
            return names
        sub = df_train[names].copy()
        corr = sub.astype(float).corr(method="pearson")
        to_drop: set = set()
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = abs(corr.iloc[i, j])
                if c > corr_threshold:
                    a, b = cols[i], cols[j]
                    miss_a = compute_missing_ratio(df_train[a])
                    miss_b = compute_missing_ratio(df_train[b])
                    drop_col = a if miss_a >= miss_b else b
                    to_drop.add(drop_col)
        kept_ordered = [c for c in names if c not in to_drop]
        for col in to_drop:
            dropped["numerical_features"][col] = "correlation"
        return kept_ordered

    numerical_final = filter_correlation(numerical_kept)
    categorical_final = categorical_kept

    selected = {
        "numerical_features": numerical_final,
        "categorical_features": categorical_final,
    }
    return selected, dropped


def save_selected_features(path: str, selected: Dict[str, List[str]]) -> None:
    """Сохраняет selected_features.json по указанному пути."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)


def save_dropped_features(path: str, dropped: Dict[str, Dict[str, str]]) -> None:
    """Сохраняет dropped_features.json по указанному пути."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dropped, f, ensure_ascii=False, indent=2)
