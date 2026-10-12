from dataclasses import dataclass

import mlflow
import mlflow.pytorch
import torch
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class Prediction:
    label: str
    confidence: float


class ArabicSentimentModel:
    def __init__(
        self,
        model_name="ArabicSentiment",
        alias="production",
        tracking_uri="http://127.0.0.1:5000",
        ):

        self.model_name = model_name
        self.alias = alias
        self.tracking_uri = tracking_uri
        self.tokenizer = None
        self.model = None
        self.max_length = None

    def load_export(self, model_dir):
        model_dir = Path(model_dir)
        metadata = json.loads(
            (model_dir / "production.json").read_text(encoding="utf-8")
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        model.eval()

        self.max_length = int(metadata["max_length"])
        self.tokenizer = tokenizer
        self.model = model

        print(
            "Loaded exported production version "
            f"{metadata['registered_model_version']}"
        )
    
    def load(self) -> None:
        model_dir = os.environ.get("MODEL_DIR")
        if model_dir:
            self.load_export(model_dir)
            return
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)

        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(
            self.model_name,
            self.alias,
        )

        # Resolve production once, then use the exact version so the
        # model and tokenizer cannot come from different promotions.
        model_uri = f"models:/{self.model_name}/{version.version}"
        run = client.get_run(version.run_id)

        tokenizer_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{version.run_id}/tokenizer",
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = mlflow.pytorch.load_model(
            model_uri,
            map_location="cpu",
        )
        model.eval()

        self.max_length = int(run.data.params["max_length"])
        self.tokenizer = tokenizer
        self.model = model

        print(
            f"Loaded {self.model_name}@{self.alias}, "
            f"version {version.version}"
        )

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            output = self.model(**inputs)

        probabilities = torch.softmax(output.logits, dim=-1)
        confidences, ids = probabilities.max(dim=-1)

        return [
            Prediction(
                label=self.model.config.id2label[class_id.item()],
                confidence=confidence.item(),
            )
            for class_id, confidence in zip(ids, confidences)
        ]