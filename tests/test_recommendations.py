from app.debugger.recommendations import RecommendationEngine


def test_recommendations():
    engine = RecommendationEngine()

    mock_checks = {
        "missing": {
            "age": 10.0,
            "salary": 55.0
        },
        "imbalance": {
            "target_column": "label",
            "ratio": 0.3
        },
        "constant": ["country"],
        "correlation": [("area", "price")]
    }

    output = engine.generate(mock_checks)

    assert "recommendations" in output
    assert len(output["recommendations"]) > 0

    for rec in output["recommendations"]:
        assert "issue_type" in rec
        assert "feature" in rec
        assert "problem" in rec
        assert "recommendation" in rec