from typing import Dict, List, Tuple
import argparse
import os
from datetime import datetime
import mlflow
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")  # Используем неинтерактивный backend для работы без GUI
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool

from src.data.validation import validate_dataset, infer_feature_types
from src.data.preprocess import select_features, save_selected_features, save_dropped_features
from src.train.search import tune_catboost_depth_lr
from src.train.utils import compute_regression_metrics, ensure_dir, save_json


def split_train_valid_test(
    df: pd.DataFrame,
    target: str | List[str],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Делит датасет на train/valid/test по долям.
    
    Args:
        df: DataFrame с данными
        target: Имя целевой колонки или список имен таргетов (для совместимости)
        ratios: Доли для train/valid/test
        seed: Seed для воспроизводимости
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, valid и test датасеты
    """
    n = len(df)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(ratios[0] * n)
    n_valid = int(ratios[1] * n)

    idx_train = idx[:n_train]
    idx_valid = idx[n_train : n_train + n_valid]
    idx_test = idx[n_train + n_valid :]

    return df.iloc[idx_train].copy(), df.iloc[idx_valid].copy(), df.iloc[idx_test].copy()


def build_pools(
    df: pd.DataFrame,
    target: str | List[str],
    selected: Dict[str, List[str]],
) -> Tuple[Pool, List[str]]:
    """Создаёт CatBoost Pool и возвращает.
    
    Args:
        df: DataFrame с данными
        target: Имя целевой колонки или список имен таргетов
        selected: Словарь с выбранными числовыми и категориальными фичами
        
    Returns:
        Pool: CatBoost Pool
    """
    feature_order = selected["numerical_features"] + selected["categorical_features"]
    X = df[feature_order].copy()
    
    if isinstance(target, str):
        y = df[target]
    else:
        y = df[target].values
    
    cat_feature_names = selected["categorical_features"]
    
    for cat_col in cat_feature_names:
        if cat_col in X.columns:
            X[cat_col] = X[cat_col].fillna("None")
    
    cat_indices = [feature_order.index(c) for c in cat_feature_names]
    return Pool(X, y, cat_features=cat_indices)


def main() -> None:
    """Точка входа: обучение модели с Optuna и логированием в MLflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str, help="Путь к CSV с данными")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Путь к configs/train_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    target = cfg["target"]
    # Поддерживаем как один таргет (строка), так и список таргетов
    if isinstance(target, str):
        targets = [target]
        is_multi_target = False
    else:
        targets = target
        is_multi_target = True
    
    artifacts_dir = cfg.get("artifacts_dir", "artifacts")
    mlruns_dir = cfg.get("mlruns_dir", "mlruns")
    seed = int(cfg.get("seed", 42))

    thresholds = cfg.get("thresholds", {})
    missing_threshold = float(thresholds.get("missing_threshold", 0.5))
    corr_threshold = float(thresholds.get("corr_threshold", 0.95))

    split = cfg.get("split", {"train": 0.7, "valid": 0.15, "test": 0.15})
    ratios = (float(split["train"]), float(split["valid"]), float(split["test"]))

    df = pd.read_csv(args.data)
    df, warnings = validate_dataset(df, targets)

    numerical_features, categorical_features = infer_feature_types(df, targets)
    train_df, valid_df, test_df = split_train_valid_test(df, targets, ratios, seed)

    selected, dropped = select_features(
        train_df,
        numerical_features,
        categorical_features,
        missing_threshold=missing_threshold,
        corr_threshold=corr_threshold,
    )

    # Генерируем имя модели на основе даты (например: catboost-regressor-20251101)
    model_name = f"catboost-regressor-{datetime.now().strftime('%Y%m%d')}"
    
    # Создаем поддиректорию для модели в artifacts_dir
    model_artifacts_dir = os.path.join(artifacts_dir, model_name)
    ensure_dir(model_artifacts_dir)

    # Логирование в MLflow
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
    experiment_name = cfg.get("experiment", model_name)
    if is_multi_target:
        experiment_name = f"{experiment_name}_multitarget"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="train_catboost"):
        mlflow.log_params(
            {
                "missing_threshold": missing_threshold,
                "corr_threshold": corr_threshold,
                "seed": seed,
                "num_targets": len(targets),
                "targets": str(targets),
            }
        )
        mlflow.log_dict(selected, artifact_file="selected_features.json")
        mlflow.log_dict(dropped, artifact_file="dropped_features.json")

        train_pool = build_pools(train_df, targets, selected)
        valid_pool = build_pools(valid_df, targets, selected)
        test_pool = build_pools(test_df, targets, selected)

        if is_multi_target:
            loss_function = "MultiRMSE"
            eval_metric = "MultiRMSE"
        else:
            loss_function = "RMSE"
            eval_metric = "RMSE"

        # Optuna
        optuna_cfg = cfg.get("optuna", {"enable": True, "n_trials": 20, "timeout": None})
        if optuna_cfg.get("enable", True):
            best_params, best_value = tune_catboost_depth_lr(
                train_pool=train_pool,
                valid_pool=valid_pool,
                n_trials=int(optuna_cfg.get("n_trials", 20)),
                timeout=optuna_cfg.get("timeout", None),
                seed=seed,
                early_stopping_rounds=int(optuna_cfg.get("early_stopping_rounds", 200)),
                loss_function=loss_function,
                eval_metric=eval_metric,
            )
            mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("optuna_valid_rmse", best_value)
        else:
            best_params = {"depth": 6, "learning_rate": 0.06}

        # Финальное обучение ТОЛЬКО на train с лучшими гиперпараметрами
        model = CatBoostRegressor(
            loss_function=loss_function,
            eval_metric=eval_metric,
            random_seed=seed,
            verbose=False,
            **best_params,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)
        
        # Логируем метрики по эпохам в MLflow и строим график
        evals_result = model.get_evals_result()
        if evals_result:
            metric_name = "MultiRMSE" if is_multi_target else "RMSE"
            train_rmse = []
            valid_rmse = []
            
            # Ищем метрики train (learn) и validation в результатах
            for dataset_name, metrics_dict in evals_result.items():
                if metric_name in metrics_dict:
                    rmse_values = metrics_dict[metric_name]
                    if isinstance(rmse_values, list) and len(rmse_values) > 0:
                        if dataset_name.lower() in ["learn", "train", "training"]:
                            train_rmse = rmse_values
                        elif dataset_name.lower() in ["validation", "valid", "eval"]:
                            valid_rmse = rmse_values
            
            # Логируем метрики по эпохам
            if train_rmse:
                for step, value in enumerate(train_rmse):
                    mlflow.log_metric("train_rmse", float(value), step=step)
            if valid_rmse:
                for step, value in enumerate(valid_rmse):
                    mlflow.log_metric("valid_rmse", float(value), step=step)
            
            if train_rmse or valid_rmse:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if train_rmse:
                    epochs = range(len(train_rmse))
                    ax.plot(epochs, train_rmse, label="Train RMSE", marker="o", markersize=3, linewidth=2, color="#1f77b4")
                
                if valid_rmse:
                    epochs = range(len(valid_rmse))
                    ax.plot(epochs, valid_rmse, label="Validation RMSE", marker="s", markersize=3, linewidth=2, color="#ff7f0e")
                
                ax.set_xlabel("Эпоха (Iteration)", fontsize=12)
                ax.set_ylabel("RMSE", fontsize=12)
                title = "Изменение метрик RMSE в процессе обучения"
                if is_multi_target:
                    title += f" ({len(targets)} таргетов)"
                ax.set_title(title, fontsize=14, fontweight="bold")
                ax.legend(fontsize=11, loc="best")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                mlflow.log_figure(fig, "training_metrics.png")
                plt.close(fig)

        y_true_all = test_pool.get_label()
        y_pred_all = model.predict(test_pool)
        
        if is_multi_target:
            # Вычисляем метрики для каждого таргета отдельно
            for idx, target_name in enumerate(targets):
                y_true = y_true_all[:, idx].tolist()
                y_pred = y_pred_all[:, idx].tolist()
                metrics = compute_regression_metrics(y_true, y_pred)
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"test_{target_name}_{metric_name}", metric_value)
            
            # Также логируем средние метрики по всем таргетам
            all_metrics = {"rmse": [], "mae": [], "r2": []}
            for idx in range(len(targets)):
                y_true = y_true_all[:, idx].tolist()
                y_pred = y_pred_all[:, idx].tolist()
                metrics = compute_regression_metrics(y_true, y_pred)
                all_metrics["rmse"].append(metrics["rmse"])
                all_metrics["mae"].append(metrics["mae"])
                all_metrics["r2"].append(metrics["r2"])
            
            # Средние метрики
            mlflow.log_metrics({
                "test_avg_rmse": float(np.mean(all_metrics["rmse"])),
                "test_avg_mae": float(np.mean(all_metrics["mae"])),
                "test_avg_r2": float(np.mean(all_metrics["r2"])),
            })
        else:
            # Для одного таргета
            y_true = y_true_all.tolist() if hasattr(y_true_all, 'tolist') else list(y_true_all)
            y_pred = y_pred_all.tolist() if hasattr(y_pred_all, 'tolist') else list(y_pred_all)
            metrics = compute_regression_metrics(y_true, y_pred)
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        # Логируем таблицу с примерами из тестовой выборки
        # Выбираем 10 случайных примеров из тестовой выборки (с seed для воспроизводимости)
        n_examples = min(10, len(test_df))
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(test_df), size=n_examples, replace=False)
        feature_order = selected["numerical_features"] + selected["categorical_features"]
        sample_features = test_df.iloc[sample_indices][feature_order].copy()
        sample_targets_true = test_df.iloc[sample_indices][targets].copy()
        sample_df = test_df.iloc[sample_indices].copy()
        sample_pool = build_pools(sample_df, targets, selected)
        sample_predictions = model.predict(sample_pool)
        
        table_data = {}
        for col in feature_order:
            values = sample_features[col].tolist()
            if col in selected["numerical_features"]:
                table_data[col] = [float(v) if pd.notna(v) else None for v in values]
            else:
                table_data[col] = [str(v) if pd.notna(v) else None for v in values]
        
        # Добавляем реальные таргеты и предсказания
        if is_multi_target:
            # Для мультитаргета
            for idx, target_name in enumerate(targets):
                true_values = sample_targets_true[target_name].tolist()
                pred_values = sample_predictions[:, idx].tolist()
                table_data[f"{target_name}_true"] = [float(v) for v in true_values]
                table_data[f"{target_name}_pred"] = [float(v) for v in pred_values]
                # Вычисляем абсолютную ошибку для каждого примера
                table_data[f"{target_name}_error"] = [
                    abs(float(true) - float(pred)) for true, pred in zip(true_values, pred_values)
                ]
        else:
            # Для одного таргета
            target_name = targets[0]
            true_values = sample_targets_true[target_name].tolist()
            if hasattr(sample_predictions, 'tolist'):
                pred_values = sample_predictions.tolist()
            elif isinstance(sample_predictions, np.ndarray):
                pred_values = sample_predictions.flatten().tolist()
            else:
                pred_values = list(sample_predictions)
            table_data[f"{target_name}_true"] = [float(v) for v in true_values]
            table_data[f"{target_name}_pred"] = [float(v) for v in pred_values]
            # Вычисляем абсолютную ошибку для каждого примера
            table_data[f"{target_name}_error"] = [
                abs(float(true) - float(pred)) for true, pred in zip(true_values, pred_values)
            ]
        
        examples_df = pd.DataFrame(table_data)
        mlflow.log_table(data=examples_df, artifact_file="test_examples.json")
        
        # Сохраняем данные тестовых примеров для использования в тестах
        test_examples_data = {
            "features": [],
            "targets_true": {},
        }
        
        # Добавляем фичи для каждого примера
        for idx in sample_indices:
            example_features = {}
            for col in feature_order:
                value = test_df.iloc[idx][col]
                if pd.notna(value):
                    example_features[col] = float(value) if col in selected["numerical_features"] else str(value)
                else:
                    example_features[col] = None
            test_examples_data["features"].append(example_features)
        
        # Добавляем реальные таргеты для каждого примера
        for target_name in targets:
            test_examples_data["targets_true"][target_name] = []
            for idx in sample_indices:
                value = test_df.iloc[idx][target_name]
                test_examples_data["targets_true"][target_name].append(float(value))
        
        # Сохраняем индексы выбранных примеров и данные для тестов
        test_examples_info = {
            "model_name": model_name,
            "endpoint": f"/{model_name}",
            "sample_indices": sample_indices.tolist(),
            "targets": targets,
            "is_multi_target": is_multi_target,
            "test_examples": test_examples_data,
        }
        
        # Сохраняем артефакты в поддиректорию с именем модели
        model_path = os.path.join(model_artifacts_dir, "model.cbm")
        sel_path = os.path.join(model_artifacts_dir, "selected_features.json")
        drop_path = os.path.join(model_artifacts_dir, "dropped_features.json")
        model_info_path = os.path.join(model_artifacts_dir, "model_info.json")

        model.save_model(model_path)
        save_selected_features(sel_path, selected)
        save_dropped_features(drop_path, dropped)
        save_json(model_info_path, test_examples_info)
        
        # Логируем имя модели в MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("endpoint", f"/{model_name}")

        # Логируем артефакты в MLflow
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(sel_path)
        mlflow.log_artifact(drop_path)
        mlflow.log_artifact(model_info_path)

        if warnings:
            mlflow.log_dict({"warnings": warnings}, artifact_file="validation_warnings.json")


if __name__ == "__main__":
    main()
