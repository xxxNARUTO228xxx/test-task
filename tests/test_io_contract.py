import os
import tempfile
from fastapi.testclient import TestClient
import pandas as pd
from catboost import CatBoostRegressor

from src.api.main import app


def _prepare_artifacts(tmpdir: str) -> None:
    df = pd.DataFrame({"num": [0, 1, 2, 3], "y": [0.0, 1.0, 2.0, 3.0]})
    m = CatBoostRegressor(loss_function="RMSE", verbose=False)
    m.fit(df[["num"]], df["y"])
    m.save_model(os.path.join(tmpdir, "model.cbm"))
    with open(os.path.join(tmpdir, "selected_features.json"), "w", encoding="utf-8") as f:
        f.write('{"numerical_features":["num"],"categorical_features":[]}')


def test_io_validation_ok_with_missing() -> None:
    with tempfile.TemporaryDirectory() as d:
        _prepare_artifacts(d)
        os.environ["ARTIFACTS_DIR"] = d
        client = TestClient(app)
        payload = {"items": [{"features": {"num": None}}]}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "predictions" in body
        assert isinstance(body["predictions"], list)


def test_io_validation_bad_schema() -> None:
    with tempfile.TemporaryDirectory() as d:
        _prepare_artifacts(d)
        os.environ["ARTIFACTS_DIR"] = d
        client = TestClient(app)
        # отсутствует ключ features
        payload = {"items": [{}]}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422
