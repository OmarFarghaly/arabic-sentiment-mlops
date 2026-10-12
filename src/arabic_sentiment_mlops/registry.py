import math

from mlflow import MlflowClient


class ModelPromoter:
    def __init__(
        self,
        tracking_uri="http://127.0.0.1:5000",
        model_name="ArabicSentiment",
        min_improvement=0.005,
    ):
        if not math.isfinite(min_improvement) or min_improvement < 0:
            raise ValueError("min_improvement must be finite and nonnegative.")

        self.client = MlflowClient(
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )
        self.model_name = model_name
        self.min_improvement = min_improvement

    def evaluation(self, version):
        if version.status != "READY":
            raise ValueError(f"Version {version.version} is not ready.")

        if not version.run_id:
            raise ValueError("Model version has no associated run.")

        run = self.client.get_run(version.run_id)

        if run.info.status != "FINISHED":
            raise ValueError(f"Run {version.run_id} has not finished successfully.")

        score = run.data.metrics.get("test_f1_macro")
        data_hash = run.data.params.get("test_data_sha256")

        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Run has no valid test_f1_macro. Evaluate it first.")

        if not data_hash:
            raise ValueError("Run has no test_data_sha256.")

        return score, data_hash

    def promote(self):
        registered = self.client.get_registered_model(self.model_name)
        aliases = registered.aliases

        if "staging" not in aliases:
            raise ValueError("No staging alias exists.")

        candidate = self.client.get_model_version(
            self.model_name,
            aliases["staging"],
        )
        candidate_score, candidate_hash = self.evaluation(candidate)

        production_version = aliases.get("production")

        if production_version == str(candidate.version):
            print("Staging is already production.")
            return False

        if production_version is None:
            # Bootstrap policy: the first successfully evaluated candidate
            # becomes production.
            reason = "First evaluated production model."
        else:
            production = self.client.get_model_version(
                self.model_name,
                production_version,
            )
            production_score, production_hash = self.evaluation(production)

            if candidate_hash != production_hash:
                raise ValueError(
                    "Test datasets differ. Promotion refused."
                )

            improvement = candidate_score - production_score

            print(
                f"Staging F1: {candidate_score:.6f}; "
                f"production F1: {production_score:.6f}; "
                f"improvement: {improvement:.6f}"
            )

            if improvement <= 0 or improvement < self.min_improvement:
                print("Required improvement not reached. Production unchanged.")
                return False

            reason = f"Macro F1 improved by {improvement:.6f}."

        self.client.set_registered_model_alias(
            name=self.model_name,
            alias="production",
            version=candidate.version,
        )

        print(
            f"Promoted version {candidate.version} to production. {reason}"
        )
        return True