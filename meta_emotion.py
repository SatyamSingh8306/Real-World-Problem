from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import torch
from datasets import Dataset, Audio

new_model = "superb/wav2vec2-large-superb-sid"
old_model = "superb/wav2vec2-base-superb-er"

def sentimentAudio(audio_file_path):

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(old_model)
    model = AutoModelForAudioClassification.from_pretrained(old_model)
    
    dataset = Dataset.from_dict({"audio": [audio_file_path]}).cast_column("audio", Audio(sampling_rate=16000))
    audio_input = dataset[0]["audio"]
    inputs = feature_extractor(audio_input["array"], sampling_rate=audio_input["sampling_rate"], return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = torch.argmax(logits, dim=-1).item()
    predicted_emotion = model.config.id2label[predicted_class_id]
    
    return predicted_emotion


if __name__ == "__main__":
    audio_file_path = "./audio_sample/angry.mp3"
    # predicted_emotion = sentimentAudio(audio_file_path=audio_file_path)
    labels = sentimentAudio(audio_file_path=audio_file_path)
    # print(f"Predicted Emotion: {predicted_emotion}")
    print("labels: ",labels)