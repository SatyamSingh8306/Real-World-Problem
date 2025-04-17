from huggingface_hub import InferenceClient
import librosa
import noisereduce as nr
from pydub import AudioSegment
import soundfile as sf
from dotenv import load_dotenv
import os

load_dotenv()

import numpy as np
from scipy.signal import butter, filtfilt
import soundfile as sf

file_path = "./audio_sample/happy.mp3"
output_path = file_path[:-4] + "_new.mp3"

audio_data, sample_rate = librosa.load(file_path, sr=None)

if len(audio_data.shape) > 1:
   audio_data = np.mean(audio_data, axis=1)
reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate)
normalized_audio = librosa.util.normalize(reduced_noise)
trimmed_audio, _ = librosa.effects.trim(normalized_audio, top_db=20)
resampled_audio = librosa.resample(trimmed_audio, orig_sr=sample_rate, target_sr=16000)
sf.write(output_path, resampled_audio, samplerate=16000)

# def apply_bandpass_filter(data, lowcut, highcut, sr, order=5):
#     nyquist = 0.5 * sr
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return filtfilt(b, a, data)

# filtered_audio = apply_bandpass_filter(time_stretched, lowcut=300, highcut=3000, sr=sample_rate)
# filtered_audio = time_stretched
# sf.write(audio_path, filtered_audio, sample_rate)

HF_API_KEY = os.environ.get("HF_API_KEY")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY,
)

output = client.audio_classification(output_path, model="chin-may/wav2vec2-audio-emotion-classification")

print(output)