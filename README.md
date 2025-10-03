
# Sentiment Analysis Pipeline

## Setup

### Option 1: Python venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### Option 2: Conda
conda create -n sentiment-env python=3.12 -y
conda activate sentiment-env
pip install -r requirements.txt

## Train
python src/train.py


## Predict

To make predictions with a trained sentiment model, use the `predict.py` script:

python src/predict.py --model models/sentiment.joblib "Your text here"

