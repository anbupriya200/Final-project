from preprocess import preprocess_data
from model import build_models
from evaluate import evaluate_models

if __name__ == "__main__":
    data = preprocess_data("CICIDS_Merged_80K.csv")

    models = build_models(data)

    evaluate_models(data, models)