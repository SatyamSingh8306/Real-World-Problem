from transformers import pipeline

# Initialize the audio classification pipeline with the model.
classifier = pipeline("audio-classification", model="superb/wav2vec2-large-superb-sid")

# Run the pipeline on an audio file.
# Replace 'path_to_audio.wav' with the actual path to your audio file.
result = classifier(r"audio_sample\happy.mp3")

print(result)
