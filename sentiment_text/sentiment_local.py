from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

# No API key needed if model is downloaded locally
def sentiment(query):
    # pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    pipe = pipeline("text-classification", model="cointegrated/rubert-tiny2-cedr-emotion-detection")
    result = pipe(query)
    return result
    # Output example: [{'label': 'negative', 'score': 0.98}]


if __name__ == "__main__":
    ans = sentiment("chal nikal yaha se ab")
    print(ans)