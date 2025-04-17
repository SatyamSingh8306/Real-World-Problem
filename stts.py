import os
from speech_to_text import speechToText
from meta_emotion import sentimentAudio
from text_to_speech import formatText, response, textToSpeechWithGTTS

# Define emotion mapping
EMOTION_MAPPING = {
    "ang": "calm",
    "sad": "empathetic",
    "hap": "positive",
    "neu": "neutral",
    # "surprised": "calm",
    # "fearful": "reassuring"
}

def process_user_audio(audio_file_path,output_audio_path ="response_audio_6.mp3"):
    """
    Process user audio to extract sentiment, generate a response, and convert it to speech.
    """
    user_text = speechToText(audio_file_path)
    print(f"User Text: {user_text}")

    detected_emotion = sentimentAudio(audio_file_path)
    print(f"Detected Emotion: {detected_emotion}")

    response_emotion = EMOTION_MAPPING.get(detected_emotion, "neutral")
    print(f"Response Emotion: {response_emotion}")

    generated_response = response(user_text, response_emotion)
    print(f"Generated Response: {generated_response}")

    formatted_text = formatText(generated_response, detected_emotion)
    print(f"Formatted Text: {formatted_text}")

    # output_audio_path = 
    textToSpeechWithGTTS(formatted_text, output_audio_path)
    print(f"Response Audio Saved: {output_audio_path}")

if __name__ == "__main__":
    
    user_audio_file = "./audio_sample/trail6.opus"
    process_user_audio(user_audio_file)