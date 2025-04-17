# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("audio-classification", model="Hatman/audio-emotion-detection")

# with open("trail.mp3","rb") as f:
#     result = pipe(f)
#     print(result)

result = pipe("angry.mp3")
print(result)