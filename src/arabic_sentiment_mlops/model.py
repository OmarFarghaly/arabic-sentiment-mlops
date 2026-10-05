from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class Prediction:
    label: str
    confidence: float


class ArabicSentimentModel:
    def __init__(self, model_name: str = "asafaya/bert-mini-arabic"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
        )

        self.model.eval()

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            output = self.model(**inputs)

        probabilities = torch.softmax(output.logits, dim=-1)
        confidences, ids = probabilities.max(dim=-1)

        labels = ["negative", "neutral", "positive"]

        return [
            Prediction(
                label=labels[class_id.item()],
                confidence=confidence.item(),
            )
            for class_id, confidence in zip(ids, confidences)
        ]