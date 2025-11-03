from fastapi.testclient import TestClient

from src.api.main import app


def test_catboost_regressor_20251103_endpoint():
    """Тест для эндпойнта /catboost-regressor-20251103 с проверкой предсказаний."""
    with TestClient(app) as client:
        payload = {
            "features": {
                "rfq_id": "10428",
                "customer_tier": "C",
                "material": "stainless",
                "thickness_mm": 4.0,
                "length_mm": 51.0,
                "width_mm": 1398.0,
                "holes_count": 6.0,
                "bends_count": 0.0,
                "weld_length_mm": 75.0,
                "cut_length_mm": "",
                "route": "",
                "tolerance": "precise",
                "surface_finish": "none",
                "coating": "none",
                "qty": 50.0,
                "due_days": 7.0,
                "engineer_score": 1.48,
                "part_weight_kg": 0.002,
                "labor_minutes_per_unit": 2.47,
                "material_cost_rub": 0.64,
                "labor_cost_rub": 19.76,
                "unit_price_rub": 32.14
            }
        }
        expected_predictions = [4.196251330636748, 51.46847374434046]
        response = client.post("/catboost-regressor-20251103", json=payload)
        response_data = response.json()
        actual_predictions = response_data["predictions"]
        for i, (actual, expected) in enumerate(zip(actual_predictions, expected_predictions)):
            assert actual == expected, \
                f"Prediction {i}: expected {expected}, got {actual}, difference {abs(actual - expected)}"
