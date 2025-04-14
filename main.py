from parler.parler_tts import ParlerTTSStreamer
from typing import Optional

def generate_sentiment_audio(
    text: str,
    emotion: str = "neutral",  # Supported emotions: "happy", "sad", "angry", etc.
    output_path: str = "output.wav",
    model_name: str = "parler-tts/parler-tts-mini-v1",  # Default model [[2]]
    pitch: float = 1.0,  # Adjust pitch (0.5 to 2.0) [[3]]
    speaking_rate: float = 1.0,  # Adjust speed (0.5 to 2.0) [[3]]
) -> None:
    """
    Generate audio with sentiment control using Parler-TTS.
    
    Args:
        text (str): Input text to convert to speech.
        emotion (str): Target emotion (e.g., "happy", "sad") [[1]].
        output_path (str): Path to save the generated WAV file.
        model_name (str): Hugging Face model name (e.g., "parler-tts/parler-tts-large-v1") [[6]].
        pitch (float): Voice pitch multiplier (lower = deeper, higher = shriller) [[3]].
        speaking_rate (float): Speech speed multiplier [[3]].
    """
    # Load pre-trained model
    model = ParlerTTSStreamer.from_pretrained(model_name)  # [[2]][[6]][[7]]
    
    # Generate audio with sentiment and style controls
    audio = model.generate(
        text,
        emotion=emotion,
        pitch=pitch,
        speaking_rate=speaking_rate
    )  # [[1]][[3]]
    
    # Save audio to file
    audio.save(output_path)
    print(f"Audio saved to {output_path}")



if __name__ == "__main__":
    generate_sentiment_audio(
    text="I can't believe we won the competition!",
    emotion="happy",
    output_path="happy_message.wav",
    pitch=1.2,  # Slightly higher-pitched voice
    speaking_rate=1.1  # Slightly faster speech
    )