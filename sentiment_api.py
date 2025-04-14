"""
Currently It is under Process.It will take time
"""

from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()


HF_API_KEY = os.getenv("HF_API_KEY")  
client = InferenceClient(token=HF_API_KEY)

# Define your query
query = "You have to leave now"


result = client.text_classification(
    model="tabularisai/multilingual-sentiment-analysis",
    text=query
)

print(result)
