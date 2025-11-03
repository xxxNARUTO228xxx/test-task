from fastapi.testclient import TestClient

from src.api.main import app


def test_tolerance_catboost_regressor_20251103_endpoint():
    """Тест для эндпойнта /catboost-regressor-20251103 с проверкой предсказаний."""
    with TestClient(app) as client:
        payload = {
            "features": {
                "rfq_id": "3686",
                "customer_tier": "B",
                "material": "steel",
                "thickness_mm": 8.13,
                "length_mm": 531.0,
                "width_mm": 153.0,
                "holes_count": 4.0,
                "bends_count": 1.0,
                "weld_length_mm": 124.0,
                "cut_length_mm": 6742.0,
                "route": "laser_cut",
                "tolerance": "precise",
                "surface_finish": "none",
                "coating": "powder",
                "qty": 2.0,
                "due_days": 21.0,
                "engineer_score": -1.04,
                "part_weight_kg": 0.005,
                "labor_minutes_per_unit": 14.03,
                "material_cost_rub": 0.64,
                "labor_cost_rub": 112.24,
                "unit_price_rub": 141.16
            }
        }
        expected_predictions = [14.03, 141.16]
        response = client.post("/catboost-regressor-20251103", json=payload)
        response_data = response.json()
        actual_predictions = response_data["predictions"]
        tolerance = [3.0, 10.0]
        for i, (actual, expected) in enumerate(zip(actual_predictions, expected_predictions)):
            assert abs(actual - expected) < tolerance[i], \
                f"Prediction {i}: expected {expected}, got {actual}, difference {abs(actual - expected)}"
