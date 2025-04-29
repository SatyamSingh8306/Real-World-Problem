import elevenlabs
from elevenlabs.client import ElevenLabs
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv,find_dotenv
from gtts import gTTS
import base64

import os 

load_dotenv(find_dotenv())

"MF4J4IDTRo0AxOO4dpFR"

Voice_ids = ["MF4J4IDTRo0AxOO4dpFR","ni6cdqyS9wBvic5LPA7M","SGbOfpm28edC83pZ9iGb","tTZ0TVc9Q1bbWngiduLK","FFmp1h1BMl0iVHA0JxrI"]

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY") 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def textToSpeechWithGTTS(input_text,output_filepath ):
    language = "en"
    audioobj = gTTS(
        text=input_text,
        lang = language,
        slow = False
    )

    audioobj.save(output_filepath)

def text_to_speech_with_elevenlabs(input_text,output_file_path ):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text = input_text,
        voice = "ni6cdqyS9wBvic5LPA7M",
        output_format="mp3_22050_32",
        model = "eleven_turbo_v2"
    )
    elevenlabs.save(audio,output_file_path)

def formatText(input_text,emotion):
    prompt = PromptTemplate(
    template="""
                Format the following text to reflect {emotion} in audio delivery while ensuring clear pronunciation, especially for English TTS systems like gTTS. Original text: "{text}"

                Instructions:
                1. NEVER change the actual words or meaning - only modify formatting for emotional expression.
                2. Focus on adjustments that work well with English text-to-speech systems:
                - Use SIMPLE punctuation that TTS can interpret clearly
                - Prioritize clarity over artistic formatting
                - Avoid complex symbols that might confuse voice synthesis

                3. Allowed formatting adjustments:
                - **Basic Punctuation**:
                    * ! for excitement (but don't overuse)
                    * ... for pauses (use sparingly)
                    * - for short breaks
                - **Capitalization**:
                    * ALL CAPS for strong emphasis (1-2 words max)
                - **Spacing**:
                    * Slight extra spaces for pacing (e.g., "wait... wait")
                - **Pronunciation Hints** (if needed):
                    * Add hyphens for tricky words (e.g., "pro-nun-ci-a-tion")

                4. Important:
                - Keep sentences short and clear
                - Avoid nested punctuation
                - Maintain natural English flow
                - Test your formatting with basic TTS systems

                Example:
                Original: "I don't understand this situation"
                Angry: "I DON'T understand this situation!"
                Confused: "I... don't understand this situation."

                RETURN only the reponse content - no explanation,no heading or somethingelse
            """,
                input_variables=["text", "emotion"]
        )


    model = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.5,
        api_key=GROQ_API_KEY
    )
    chain = prompt | model
    text = chain.invoke({"emotion":emotion,"text":input_text}).content
    return text


def response(input_text,emotion):
    prompt = PromptTemplate(
    template="""Act as an expert Customer Care Agent and a common man. Generate a response to the customer's message: '{text}' 
    with the following requirements:
    1. Express {emotion} through tone and word choice (e.g., empathy for frustration, enthusiasm for positive interactions)
    2. Address the core issue using probing questions if needed (e.g., 'Could you share the specific error message?' [[4]])
    3. Include actionable solutions while avoiding technical jargon 
    4. Offer clear next steps or escalation paths if unresolved 
    5. Maintain concise, professional language without markdown formatting

    NOTE: make sure that reponse should be concise and 
    RETURN only the reponse content - no explanation,no heading or somethingelse""",
    input_variables=["text", "emotion"]
)
    model = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.5,
        api_key=GROQ_API_KEY
    )
    chain = prompt | model
    text = chain.invoke({"emotion":emotion,"text":input_text}).content
    return text


if __name__ =="__main__":
    text = "Mai ye nahi kar sakti vo meri dost hai"
    text = formatText(text)
    # text_to_speech_with_elevenlabs(text,"audio7.mp3")
    textToSpeechWithGTTS(text,output_filepath="audio7.mp3")
    