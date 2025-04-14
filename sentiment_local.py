from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

# No API key needed if model is downloaded locally
def sentiment(query):
    pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    result = pipe(query)
    return result[0]["label"]
    # Output example: [{'label': 'negative', 'score': 0.98}]


if __name__ == "__main__":
    ans = sentiment("Hi how are you doing")
    print(ans)