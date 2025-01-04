import openai
from elevenlabs import generate, play, set_api_key, Voice
import speech_recognition as sr
import pyaudio
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

os.environ["PATH"] += ";C:\\ffmpeg\\bin"

# Set up API keys from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
set_api_key(elevenlabs_key)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def detect_wake_word(text):
    wake_word_detected = "jarvis" in text.lower()
    if wake_word_detected:
        print("Wake word detected!")
        response_audio = generate(
            text="How can I help you?",
            voice=Voice(voice_id="EXAVITQu4vr4xnSDxMaL")
        )
        play(response_audio)
    return wake_word_detected

# Add this function to get weather data
def get_weather(location):
    try:
        # Get coordinates for location using geocoding API
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_response = requests.get(geocode_url)
        geo_data = geo_response.json()
        
        if not geo_data.get('results'):
            return f"Sorry, I couldn't find the location: {location}"
            
        # Extract coordinates
        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        
        # Get weather data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&temperature_unit=fahrenheit"
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()
        temp = weather_data['current_weather']['temperature']
        
        return f"The current temperature in {location} is {temp}°F"
    except Exception as e:
        return f"Sorry, I couldn't fetch the weather information: {str(e)}"

# Modify the get_chatgpt_response function
def get_chatgpt_response(prompt):
    # Check if prompt is asking about weather
    if "weather" in prompt.lower():
        # Extract location from prompt
        words = prompt.lower().split()
        try:
            loc_index = words.index("in") + 1
            location = " ".join(words[loc_index:])
            return get_weather(location)
        except ValueError:
            return "Please specify a location. For example: 'What's the weather in London?'"
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def synthesize_speech(text):
    try:
        print("Generating audio...")
        audio = generate(
            text=text,
            voice="Chris",
            model="eleven_monolingual_v1"
        )
        
        print("Playing audio...")
        play(audio)
        
    except Exception as e:
        print(f"Error in speech synthesis: {str(e)}")

def process_user_question(question):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}]
        )
        answer = response.choices[0].message.content
        print(f"AI response: {answer}")
        
        response_audio = generate(
            text=answer,
            voice=Voice(voice_id="EXAVITQu4vr4xnSDxMaL"),
            api_key=elevenlabs_key
        )
        play(response_audio)
    except Exception as e:
        print(f"Error processing question: {e}")

def main():
    r = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Calibrating for ambient noise... Please wait")
        r.adjust_for_ambient_noise(source, duration=2)
        r.energy_threshold = 4000
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.8
        
        while True:
            try:
                print("Listening...")
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
                
                print("Processing speech...")
                text = r.recognize_google(audio)
                print(f"You said: {text}")
                
                if "jarvis" in text.lower():
                    response_audio = generate(
                        text="How can I help you?",
                        voice=Voice(voice_id="EXAVITQu4vr4xnSDxMaL"),
                        api_key=elevenlabs_key
                    )
                    play(response_audio)
                    
                    print("Listening for your question...")
                    question_audio = r.listen(source, timeout=5, phrase_time_limit=10)
                    question_text = r.recognize_google(question_audio)
                    print(f"Your question: {question_text}")
                    
                    # Process and respond to the question
                    process_user_question(question_text)
                    
            except sr.WaitTimeoutError:
                print("Listening timed out. Please try again.")
            except sr.UnknownValueError:
                print("Could not understand audio. Please speak more clearly.")
            except sr.RequestError as e:
                print(f"Could not process audio; {e}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
