import argparse
from typing import Any, Sequence, Tuple, List
import numpy as np
from numpy.typing import NDArray
from joblib import load


def load_model(model_path: str) -> Any:
    """Load and return a trained classifier."""
    return load(model_path)


def predict_texts(
    classifier: Any, input_texts: Sequence[str]
) -> Tuple[List[str], List[float | None]]:
    """Return labels (str) and probability-of-positive for each text."""
    preds: list[str] = classifier.predict(input_texts).tolist()

    if hasattr(classifier, "predict_proba"):
        probs_arr: NDArray[np.float64] = classifier.predict_proba(input_texts)[:, 1]
        probs: list[float | None] = [float(p) for p in probs_arr.tolist()]
    else:
        probs: list[float | None] = [None] * len(input_texts)

    return preds, probs


def format_prediction_lines(
    texts: Sequence[str], preds: Sequence[str], probs: Sequence[float | None]
) -> List[str]:
    """Return tab-separated CLI output lines for each input text."""
    lines: List[str] = []
    for text, pred, prob in zip(texts, preds, probs):
        if prob is None:
            lines.append(f"{pred}\t{text}")
        else:
            lines.append(f"{pred}\t{prob:.3f}\t{text}")
    return lines


def main(model_path: str, input_texts: Sequence[str]) -> None:
    classifier = load_model(model_path)
    preds, probs = predict_texts(classifier, input_texts)
    for line in format_prediction_lines(input_texts, preds, probs):
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict sentiment using a trained model"
    )
    parser.add_argument(
        "--model", default="models/sentiment.joblib", help="Path to trained model"
    )
    parser.add_argument("text", nargs="+", help="One or more texts to score")
    args = parser.parse_args()
    main(model_path=args.model, input_texts=args.text)
