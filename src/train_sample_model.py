import logging
import os

from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

# Sample data
sample_texts = ["I love this movie!", "I hate this movie!"]
sample_labels = [1, 0]

# Build a simple pipeline
pipeline = make_pipeline(
    CountVectorizer(),
    MultinomialNB(),
)

# Train the model
pipeline.fit(sample_texts, sample_labels)

# Save the model
model_path = "models/sample_model.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
dump(pipeline, model_path)
logging.info(f"Saved model to {model_path}")
