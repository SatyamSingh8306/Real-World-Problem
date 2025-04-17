# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")

result = pipe("./audio_sample/angry.mp3")
print(result)