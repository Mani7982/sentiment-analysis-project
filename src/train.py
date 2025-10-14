import logging
import os

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)


def load_and_validate_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"],
            df["label"],
            test_size=0.2,
            random_state=42,
            stratify=df["label"],
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"],
            df["label"],
            test_size=0.2,
            random_state=42,
        )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    clf_pipeline = make_pipeline(
        TfidfVectorizer(min_df=1, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000),
    )
    clf_pipeline.fit(X_train, y_train)
    return clf_pipeline


def save_model(model: Pipeline, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)
    logging.info(f"Saved model to {model_path}")


if __name__ == "__main__":
    df = load_and_validate_data("sentiments.csv")
    logging.info("Loaded data:")
    logging.info(df.head())

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model, "models/sentiment_model.pkl")
