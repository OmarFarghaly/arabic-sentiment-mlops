For running MLflow: uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
For Training: uv run dvc repro train
Run App: uv run uvicorn arabic_sentiment_mlops.api:app --host 127.0.0.1 --port 8000



Flow of Training/evaluate Model:
Train → register model → staging
Evaluate → log test results to its run
Promote → compare staging with production
API restart → load production model









Training run
  └─ saved model + parameters + validation metrics
       └─ evaluation adds test_f1_macro to that run
            └─ register that model as ArabicSentiment, version 1
                 └─ staging points to version 1
                      └─ promote.py checks its score
                           └─ production points to the approved version
                                └─ API loads it on startup