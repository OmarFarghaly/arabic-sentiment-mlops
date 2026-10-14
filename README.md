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





After downloading the tar.file:
     1. docker load --input "C:\path\to\arabic-sentiment-api.tar"
     2. docker compose up
     3. Invoke-RestMethod http://localhost:8000/health --> Test
     




To run the load Test on the App using locust:
     New-Item -ItemType Directory -Force reports | Out-Null
     uv run locust --config locust.conf

To see the results:
     Start-Process reports/locust_fastapi.html