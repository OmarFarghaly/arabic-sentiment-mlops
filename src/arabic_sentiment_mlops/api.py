from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from arabic_sentiment_mlops.model import ArabicSentimentModel


model = ArabicSentimentModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load()
    yield


app = FastAPI(
    title="Arabic Sentiment API",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    prediction = model.predict(request.text)

    return {
        "label": prediction.label,
        "confidence": prediction.confidence,
    }


@app.post("/predict_batch")
def predict_batch(request: BatchPredictRequest):
    predictions = model.predict_batch(request.texts)

    return {
        "predictions": [
            {
                "label": prediction.label,
                "confidence": prediction.confidence,
            }
            for prediction in predictions
        ]
    }