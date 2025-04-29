# Use a pipeline as a high-level helper
from transformers import pipeline
emotion_to_label = {
    "angry":0,
    "apologetic": 1,
    "base": 2,
    "calm":3,
    "excited":4,
    "fear": 5,
    "happy": 6,
    "sad": 7,
    "surprise":8 
}

def emotionInAudio(file_path):

    pipe = pipeline("audio-classification", model="aicinema69/audio-emotion-detector-try2")
    result = pipe(file_path)
    sorted_data = sorted(result, key=lambda x: x['score'], reverse=True)
    return sorted_data

if __name__ == "__main__":
    result = emotionInAudio("./audio_sample/suprise.mp3")
    print(result)