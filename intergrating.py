from text_to_speech import text_to_speech_with_gtts
from speech_to_text import speechToText
from text_to_speech import formatText

if __name__ == "__main__":
    query = "Hello this is Satyam Singh"
    text = formatText(query)
    text_to_speech_with_gtts(text,"audio.mp3")