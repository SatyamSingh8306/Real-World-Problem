import elevenlabs
from elevenlabs.client import ElevenLabs
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os 

load_dotenv()

Voice_ids = ["MF4J4IDTRo0AxOO4dpFR","ni6cdqyS9wBvic5LPA7M","SGbOfpm28edC83pZ9iGb","tTZ0TVc9Q1bbWngiduLK","FFmp1h1BMl0iVHA0JxrI"]

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY") 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def text_to_speech_with_elevenlabs(input_text,output_file_path ):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text = input_text,
        voice = "FFmp1h1BMl0iVHA0JxrI",
        output_format="mp3_22050_32",
        model = "eleven_turbo_v2"
    )
    elevenlabs.save(audio,output_file_path)

def formatText(input_text):
    prompt = PromptTemplate(
            template="""Format the following text to reflect {emotion} in audio delivery for ElevenLabs, without changing the original text content:"{text}"

            Instructions:
            1. DO NOT change, paraphrase, or add to the text content.
            2. Only modify punctuation, capitalization, spacing (e.g., ellipses "...", exclamation marks "!", pauses), and optionally suggest voice pacing notes (like speed, pitch) to convey the intended emotion.
            3. Avoid adding narrative context or dialogue tags.
            4. Ensure the formatting makes the emotional tone clear for text-to-speech conversion.

            Example:
            Input text: "I can't believe this"
            Formatted for anger: "I CAN'T BELIEVE THIS!!!"
            Formatted for sadness: "I... can't believe this..."

            NOTE: Do not alter the wording — only adjust formatting for emotional expression.

            Return only the formatted text.""",
                input_variables=["text", "emotion"]
            )

    model = ChatGroq(
        model="qwen-2.5-32b",
        temperature=0.8
    )
    chain = prompt | model
    text = chain.invoke({"emotion":"sadness","text":input_text}).content
    return text


if __name__ =="__main__":
    text = "Mai ye nahi kar sakti vo meri dost hai"
    text = formatText(text)
    text_to_speech_with_elevenlabs(text,"audio6.mp3")