import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentEvaluator:
    def __init__(self, params_path="params.yaml"):
        with open(params_path, encoding="utf-8") as file:
            params = yaml.safe_load(file)

        self.model_dir = Path(params["train"]["model_output_dir"])
        self.test_path = Path(params["prepare"]["output_dir"]) / "test.csv"
        self.metrics_path = Path(params["evaluate"]["metrics_path"])

        self.metadata = json.loads(
            (self.model_dir / "mlflow_run.json").read_text(
                encoding="utf-8"
            )
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        ).to(self.device)
        self.model.eval()

    def load_test_data(self):
        frame = pd.read_csv(self.test_path, encoding="utf-8-sig")

        if frame.empty:
            raise ValueError("Test data is empty.")

        if frame[["text", "label"]].isna().any().any():
            raise ValueError("Test data contains missing texts or labels.")

        if frame["text"].str.strip().eq("").any():
            raise ValueError("Test data contains blank texts.")

        label_to_id = self.model.config.label2id

        if not set(frame["label"]).issubset(label_to_id):
            raise ValueError("Test data contains unknown labels.")

        return (
            frame["text"].tolist(),
            frame["label"].map(label_to_id).to_numpy(),
        )

    def predict(self, texts):
        predictions = []
        batch_size = self.metadata["batch_size"]

        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer(
                    texts[start:start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.metadata["max_length"],
                    return_tensors="pt",
                ).to(self.device)

                logits = self.model(**inputs).logits
                predictions.extend(
                    logits.argmax(dim=-1).cpu().tolist()
                )

        return predictions

    def run(self):
        mlflow.set_tracking_uri(self.metadata["tracking_uri"])

        # Resume the training run instead of creating another run.
        with mlflow.start_run(run_id=self.metadata["run_id"]):
            self.load()
            texts, true_labels = self.load_test_data()
            predictions = self.predict(texts)

            label_ids = sorted(
                int(label_id)
                for label_id in self.model.config.id2label
            )
            label_names = [
                self.model.config.id2label[label_id]
                for label_id in label_ids
            ]

            metrics = {
                "test_accuracy": float(
                    accuracy_score(true_labels, predictions)
                ),
                "test_f1_macro": float(
                    f1_score(
                        true_labels,
                        predictions,
                        labels=label_ids,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }

            mlflow.log_params({
                "test_rows": len(texts),
                "test_data_sha256": hashlib.sha256(
                    self.test_path.read_bytes()
                ).hexdigest(),
            })
            mlflow.log_metrics(metrics)

            self.metrics_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.metrics_path.write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(
                str(self.metrics_path),
                artifact_path="evaluation",
            )
            mlflow.log_artifact(
                __file__,
                artifact_path="reproducibility",
            )

            with TemporaryDirectory() as temporary_dir:
                display = ConfusionMatrixDisplay.from_predictions(
                    true_labels,
                    predictions,
                    labels=label_ids,
                    display_labels=label_names,
                    cmap="Blues",
                    colorbar=False,
                )
                display.ax_.set_title("Held-out test confusion matrix")
                display.figure_.tight_layout()

                image_path = (
                    Path(temporary_dir) / "confusion_matrix.png"
                )
                display.figure_.savefig(image_path, dpi=150)
                plt.close(display.figure_)
                mlflow.log_artifact(str(image_path))

            print(f"MLflow run ID: {self.metadata['run_id']}")
            print(metrics)


if __name__ == "__main__":
    SentimentEvaluator().run()