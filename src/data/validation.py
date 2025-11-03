from typing import Dict, List, Optional, Tuple
import pandas as pd


def validate_dataset(
    df: pd.DataFrame,
    target_column: str | List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Валидирует входной датасет и возвращает (df, warnings).

    Args:
        df: DataFrame с данными
        target_column: Имя целевой колонки или список имен таргетов

    Returns:
        Tuple[pd.DataFrame, List[str]]: Валидированный DataFrame и список предупреждений

    Проверки:
    - наличие целевой колонки
    - дубликаты строк (предупреждение)
    - пустые строки целевой переменной (удаляются)
    """
    warnings: List[str] = []

    # Поддерживаем как один таргет, так и список таргетов
    if isinstance(target_column, str):
        target_columns = [target_column]
    else:
        target_columns = target_column

    # Проверяем наличие всех таргетов
    missing_targets = [t for t in target_columns if t not in df.columns]
    if missing_targets:
        raise ValueError(f"Не найдены целевые колонки: {missing_targets}")

    if df.duplicated().any():
        warnings.append("Обнаружены дубликаты строк")

    # Удаляем строки, где хотя бы один таргет пустой
    before = len(df)
    # Строка удаляется, если хотя бы один таргет имеет NaN
    mask = df[target_columns].notna().all(axis=1)
    df = df[mask].copy()
    removed = before - len(df)
    if removed > 0:
        warnings.append(f"Удалено строк с пустыми таргетами: {removed}")

    return df, warnings


def infer_feature_types(df: pd.DataFrame, target_column: str | List[str]) -> Tuple[List[str], List[str]]:
    """Определяет списки числовых и категориальных фичей по dtypes, исключая target(ы).

    Args:
        df: DataFrame с данными
        target_column: Имя целевой колонки или список имен таргетов

    Returns:
        Tuple[List[str], List[str]]: Списки числовых и категориальных фичей
    """
    if isinstance(target_column, str):
        target_columns = [target_column]
    else:
        target_columns = target_column

    # Исключаем все таргеты из списка фичей
    feature_columns = [c for c in df.columns if c not in target_columns]
    numerical_features: List[str] = []
    categorical_features: List[str] = []

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_features.append(col)
        else:
            categorical_features.append(col)

    return numerical_features, categorical_features
