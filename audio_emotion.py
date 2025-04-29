import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Load the trained model
model = load_model('./model/model1.h5')


classes =  ['angry', 'apologetic', 'base', 'calm', 'excited', 'fear', 'happy','sad']


def preprocess_audio(file_path, sample_rate=22050, duration=3, n_mfcc=40, max_pad_len=130):
    
    audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration, res_type='kaiser_fast')
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Pad or truncate MFCCs to fixed length
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    # Add channel dimension
    mfccs = mfccs[np.newaxis, ..., np.newaxis]
    return mfccs

def predict_emotion(file_path):
    
    processed_audio = preprocess_audio(file_path)
    prediction = model.predict(processed_audio)
    predicted_index = np.argmax(prediction, axis=1)
    predicted_emotion = classes[predicted_index[0]]
    return predicted_emotion

if __name__ == "__main__":

    base_path = r"C:\Users\hp\Downloads\audio-hindi-emotion-data"
    audio_files = []
    predicted_emotions = []
    actual_emotions = []

    # Traverse through the directory structure
    for emotion_folder in os.listdir(base_path):
        emotion_folder_path = os.path.join(base_path, emotion_folder)
        
        if os.path.isdir(emotion_folder_path):  # Ensure it's a folder
            actual_emotion = emotion_folder  # Folder name represents the actual emotion
            
            for audio_file in os.listdir(emotion_folder_path):
                audio_file_path = os.path.join(emotion_folder_path, audio_file)
                
                if audio_file.endswith(".m4a"):  # Process only .wav files
                    try:
                        # Predict emotion
                        emotion = predict_emotion(audio_file_path)
                        print(f'File: {audio_file_path} Predicted Emotion: {emotion}')
                        
                        # Append results
                        audio_files.append(audio_file)
                        predicted_emotions.append(emotion)
                        actual_emotions.append(actual_emotion)
                    except Exception as e:
                        print(f"Error processing file {audio_file_path}: {e}")

    # Create a DataFrame
    df = pd.DataFrame({
        "audio files": audio_files,
        "model": ["model1 nn" for _ in range(len(audio_files))],
        "actual emotion": actual_emotions,
        "predicted emotion": predicted_emotions
    })

    # Save the DataFrame to an Excel file
    output_file = "new1.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")
    