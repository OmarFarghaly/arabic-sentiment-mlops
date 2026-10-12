import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import mlflow
import mlflow.pytorch
from transformers import AutoTokenizer


class ProductionExporter:
    def __init__(
        self,
        tracking_uri="http://127.0.0.1:5000",
        model_name="ArabicSentiment",
    ):
        self.tracking_uri = tracking_uri
        self.model_name = model_name

    def export(self, output_dir="models/production"):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)

        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(
            self.model_name,
            "production",
        )
        run = client.get_run(version.run_id)

        model = mlflow.pytorch.load_model(
            f"models:/{self.model_name}/{version.version}",
            map_location="cpu",
        )
        tokenizer_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{version.run_id}/tokenizer",
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        target = Path(output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory(dir=target.parent) as temporary_dir:
            bundle = Path(temporary_dir) / "bundle"

            model.save_pretrained(bundle, safe_serialization=True)
            tokenizer.save_pretrained(bundle)

            metadata = {
                "registered_model_name": self.model_name,
                "registered_model_version": str(version.version),
                "run_id": version.run_id,
                "max_length": int(run.data.params["max_length"]),
            }
            (bundle / "production.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

            backup = None
            if target.exists():
                backup = target.with_name(
                    f"{target.name}.backup-{uuid4().hex}"
                )
                target.rename(backup)

            try:
                bundle.rename(target)
            except Exception:
                if backup is not None:
                    backup.rename(target)
                raise

        print(f"Exported production version {version.version} to {target}")