import os
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
file_path = ""

def speechToText(file_path):
    client = Groq()
    filename = file_path

    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),
        model="distil-whisper-large-v3-en",
        response_format="verbose_json",
        )
    return transcription.text


if __name__== "__main__":
    ans = speechToText(file_path=file_path)
    print(ans)


      