from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from joblib import dump
import os

# Sample data
texts = ["I love this product", "This is amazing", "I hate this", "This is terrible"]
labels = ["positive", "positive", "negative", "negative"]

# Create a simple pipeline: vectorizer + Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(texts, labels)

# Make sure models folder exists
os.makedirs("models", exist_ok=True)

# Save the trained model
dump(model, "models/sentiment.joblib")

print("Model trained and saved to models/sentiment.joblib")
