from arabic_sentiment_mlops.export import ProductionExporter
from arabic_sentiment_mlops.registry import ModelPromoter


if __name__ == "__main__":
    ModelPromoter(min_improvement=0.005).promote()
    ProductionExporter().export()

    #Here, 0.005 means 0.5 percentage points of absolute macro F1 improvement. With no production model yet, the first successfully evaluated candidate becomes production.