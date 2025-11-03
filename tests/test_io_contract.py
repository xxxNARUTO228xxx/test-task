from fastapi.testclient import TestClient

from src.api.main import app


payload = {
    "features": {
        "rfq_id": None,
        "customer_tier": None,
        "material": None,
        "thickness_mm": None,
        "length_mm": None,
        "width_mm": None,
        "holes_count": None,
        "bends_count": None,
        "weld_length_mm": None,
        "cut_length_mm": None,
        "route": None,
        "tolerance": None,
        "surface_finish": None,
        "coating": None,
        "qty": None,
        "due_days": None,
        "engineer_score": None,
        "part_weight_kg": None,
        "labor_minutes_per_unit": None,
        "material_cost_rub": None,
        "labor_cost_rub": None,
        "unit_price_rub": None,
    }
}
expected_predictions = [1, 2]


def test_io_validation_ok_with_missing() -> None:
    with TestClient(app) as client:
        client = TestClient(app)
        response = client.post("/catboost-regressor-20251103", json=payload)
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
        response_data = response.json()
        assert "request_id" in response_data
        assert "timestamp" in response_data
        assert "predictions" in response_data
        assert isinstance(response_data["predictions"], list)
        actual_predictions = response_data["predictions"]
        assert len(actual_predictions) == len(expected_predictions), \
            f"Number of predictions does not match: expected {len(expected_predictions)}, got {len(actual_predictions)}"


def test_io_validation_bad_schema() -> None:
    with TestClient(app) as client:
        # отсутствует ключ features
        payload = {"items": [{}]}
        response = client.post("/catboost-regressor-20251103", json=payload)
        assert response.status_code == 422, f"Expected status code 422, got {response.status_code}: {response.text}"
