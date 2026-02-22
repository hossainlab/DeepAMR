"""Inference module tests for DeepAMR."""

import pytest
import numpy as np
from pathlib import Path


def test_model_version():
    from src.ml.inference import MODEL_VERSION
    assert MODEL_VERSION == "1.0.0"


@pytest.mark.skipif(
    not Path("models/advanced_deepamr_system.pt").exists(),
    reason="Model file not available",
)
class TestDeepAMRPredictor:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.ml.inference import DeepAMRPredictor
        self.predictor = DeepAMRPredictor()

    def test_model_loads(self):
        assert self.predictor.model is not None
        assert len(self.predictor.drug_classes) == 11

    def test_model_info(self):
        info = self.predictor.model_info
        assert "drug_classes" in info
        assert "device" in info
        assert "model_version" in info

    def test_predict_format(self):
        features = np.random.rand(500).astype(np.float32)
        result = self.predictor.predict(features)
        assert "predictions" in result
        assert "probabilities" in result
        assert "resistant_count" in result
        assert "susceptible_count" in result
        assert len(result["predictions"]) == 11
        assert result["resistant_count"] + result["susceptible_count"] == 11

    def test_predict_batch(self):
        features = [np.random.rand(500).astype(np.float32) for _ in range(3)]
        results = self.predictor.predict_batch(features)
        assert len(results) == 3

    def test_predict_probabilities_range(self):
        features = np.random.rand(500).astype(np.float32)
        result = self.predictor.predict(features)
        for prob in result["probabilities"].values():
            assert 0.0 <= prob <= 1.0


@pytest.mark.skipif(
    not Path("models/advanced_deepamr_system.pt").exists(),
    reason="Model file not available",
)
def test_get_predictor():
    from src.ml.inference import get_predictor
    predictor = get_predictor("deep_learning")
    assert predictor is not None


def test_get_predictor_unknown():
    from src.ml.inference import get_predictor
    with pytest.raises(ValueError):
        get_predictor("unknown_model_type")
