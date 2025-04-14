import elevenlabs
from elevenlabs.client import ElevenLabs
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os 

load_dotenv()

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY") 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def text_to_speech_with_elevenlabs(input_text,output_file_path ):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text = input_text,
        voice = "Aria",
        output_format="mp3_22050_32",
        model = "eleven_turbo_v2"
    )
    elevenlabs.save(audio,output_file_path)


if __name__ =="__main__":
    text = "Hello this is Satyam Singh"

    prompt = PromptTemplate(
        template="""Format the following text to express {emotion} (e.g., anger, joy, sadness):  
        "{text}"  

        Instructions:  
        1. Add explicit emotional descriptors in dialogue tags (e.g., 'she snapped angrily').  
        2. Include narrative context to set the tone (e.g., 'His fists clenched as he growled...').  
        3. Use punctuation to emphasize delivery (e.g., exclamation marks, ellipses).  
        4. Suggest optional voice settings (speed, pitch) for ElevenLabs.  

        Example format:  
        '[Narrative context] "[Dialogue]" [Dialogue tag with emotion], [voice pacing note].'  

        Avoid markdown. Ensure clarity for text-to-speech conversion [[5]][[3]].""",  
        input_variables=["text", "emotion"]  
    )
    model = ChatGroq(
        model="qwen-2.5-32b"
    )
    chain = prompt | model
    text = chain.invoke({"emotion":"happy","text":text}).content
    text_to_speech_with_elevenlabs(text,"audio1.mp3")