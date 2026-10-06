import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.pytorch
import transformers


class ExperimentTracker:
    def __init__(
        self,
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="arabic-sentiment",
    ):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self):
        return mlflow.start_run()

    def log_params(self, parameters):
        mlflow.log_params(parameters)

    def log_validation(self, metrics):
        mlflow.log_metrics({
            "best_val_loss": float(metrics["eval_loss"]),
            "best_val_accuracy": float(metrics["eval_accuracy"]),
            "best_val_f1_macro": float(metrics["eval_f1_macro"]),
        })

    def log_model(self, model, tokenizer, output_dir, max_length, batch_size):
        run = mlflow.active_run()
        if run is None:
            raise RuntimeError("Model logging requires an active MLflow run.")

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model.cpu(),
            name="model",
            extra_pip_requirements=[
                f"transformers=={transformers.__version__}",
            ],
        )

        with TemporaryDirectory() as temporary_dir:
            tokenizer.save_pretrained(temporary_dir)
            mlflow.log_artifacts(
                temporary_dir,
                artifact_path="tokenizer",
            )

        metadata = {
            "run_id": run.info.run_id,
            "tracking_uri": self.tracking_uri,
            "model_uri": model_info.model_uri,
            "max_length": max_length,
            "batch_size": batch_size,
        }

        metadata_path = Path(output_dir) / "mlflow_run.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(
            str(metadata_path),
            artifact_path="reproducibility",
        )

        for filename in [
            "params.yaml",
            "pyproject.toml",
            "uv.lock",
            "src/arabic_sentiment_mlops/train.py",
            "src/arabic_sentiment_mlops/callbacks.py",
            "src/arabic_sentiment_mlops/tracking.py",
        ]:
            mlflow.log_artifact(
                filename,
                artifact_path="reproducibility",
            )