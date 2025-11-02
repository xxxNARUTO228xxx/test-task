from typing import Dict, Tuple
import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostRegressor, Pool


def tune_catboost_depth_lr(
    train_pool: Pool,
    valid_pool: Pool,
    n_trials: int = 20,
    timeout: int | None = None,
    seed: int = 42,
    early_stopping_rounds: int = 200,
    loss_function: str = "RMSE",
    eval_metric: str = "RMSE",
) -> Tuple[Dict, float]:
    """Подбирает только `depth` и `learning_rate` по метрике на valid.

    Args:
        train_pool: Pool с обучающими данными
        valid_pool: Pool с валидационными данными
        n_trials: Количество trials для Optuna
        timeout: Максимальное время в секундах
        seed: Seed для воспроизводимости
        early_stopping_rounds: Количество раундов для early stopping
        loss_function: RMSE или MultiRMSE
        eval_metric: RMSE или MultiRMSE

    Returns:
        Tuple[Dict, float]: Лучшие параметры и лучшее значение метрики
    """

    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("depth", 4, 10)
        lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        model = CatBoostRegressor(
            depth=depth,
            learning_rate=lr,
            loss_function=loss_function,
            eval_metric=eval_metric,
            random_seed=seed,
            verbose=False,
        )
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        best_score = model.get_best_score()["validation"][eval_metric]
        return best_score

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_value = study.best_value
    return best_params, float(best_value)
