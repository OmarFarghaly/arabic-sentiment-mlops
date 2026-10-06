import mlflow
from transformers import TrainerCallback


class MLflowEpochCallback(TrainerCallback):
    """Log validation metrics to the active MLflow run after each epoch."""

    METRIC_NAMES = {
        "eval_loss": "val_loss",
        "eval_accuracy": "val_accuracy",
        "eval_f1_macro": "val_f1_macro",
    }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or not state.is_world_process_zero:
            return

        mlflow.log_metrics(
            {
                target: float(metrics[source])
                for source, target in self.METRIC_NAMES.items()
                if source in metrics
            },
            step=int(round(state.epoch or 0)),
        )