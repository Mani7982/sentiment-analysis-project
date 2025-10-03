# Sentiment Analysis Project

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
