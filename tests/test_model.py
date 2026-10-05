import pytest
import torch
from arabic_sentiment_mlops.model import SentimentModel

@pytest.fixture(scope="module")
def loaded_model():
    """Fixture to load the model once for all tests in this file."""
    model = SentimentModel()
    model.load()
    return model

def test_model_initialization():
    model = SentimentModel()
    assert model.model is None
    assert model.tokenizer is None

def test_predict_empty_string_raises_error(loaded_model):
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        loaded_model.predict("   ")

def test_predict_batch_returns_valid_structure(loaded_model):
    texts = [
        "المنتج ممتااااز جدا والتوصيل سريع للغاية",
        "الخدمة سيئة للغاية ولن أتعامل معهم مرة أخرى"
    ]
    results = loaded_model.predict_batch(texts)
    
    assert len(results) == 2
    for item in results:
        assert "label" in item
        assert "confidence" in item
        assert item["label"] in ["Positive", "Negative", "Neutral"]
        assert 0.0 <= item["confidence"] <= 1.0

def test_unloaded_model_raises_runtime_error():
    model = SentimentModel()
    with pytest.raises(RuntimeError, match="Model is not loaded"):
        model.predict("نص تجريبي")