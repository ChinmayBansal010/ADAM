import random
import pyttsx3
import torch
import json
import os
import re
import webbrowser
from bs4 import BeautifulSoup
import requests
import datetime
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from threading import Thread, Timer
import subprocess
import pygame
import pyautogui
import psutil
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import screen_brightness_control as sbc
from queue import Queue
import ctypes
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import winsound

#* CONSTANTS

engine = pyttsx3.init()
wake_word = "hello bot"
vosk_model = Model("model") 

speech_queue = Queue()

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'model.pth'
data = torch.load(FILE, weights_only=True)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = 'ChatBot'

stop_listening = False

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)


#* EXTRACTION FUNCTIONS
def extract_alarm_time(message):

    pattern = r"(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.|am|pm)?"
    match = re.search(pattern, message, re.IGNORECASE)

    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        ampm = match.group(3) 

        if ampm:
            if "p.m." in ampm.lower() and hour != 12:
                hour += 12
            elif "a.m." in ampm.lower() and hour == 12:
                hour = 0  
        return f"{hour}:{minute:02d} {'AM' if 'a' in ampm.lower() else 'PM'}"
    
    return None 

def extract_google_query(message):
    keywords = [
        r"search google for",
        r"google search for",
        r"search for",
        r"search google",
        r"google search",
        r"google for",
        r"can you google",
        r"could you search",
        r"please google",
        r"find information on",
        r"look up",
        r"search",
        r"google",
        r"find",
        r"can you look up",
        r"could you look up",
        r"i need info on",
        r"i want to know about",
        r"tell me about",
        r"can you find me",
        r"what is",
        r"who is",
        r"what are",
        r"how to",
        r"how do i"
    ]
    
    msg = message.lower()
    msg = re.sub(r'[^\w\s]', '', msg)

    for kw in keywords:
        pattern = rf"{kw}\s+(.*)"
        match = re.search(pattern, msg)
        if match:
            return match.group(1).strip()
    
    return msg
        
#* TASK FUNCTIONS
def mute_volume():
    volume.SetMute(1, None)  # 1 = mute

def unmute_volume():
    volume.SetMute(0, None)  # 0 = unmute

def is_muted():
    return volume.GetMute() == 1

def take_screenshot():
    try:
        screenshot = pyautogui.screenshot()
        screenshot.save("screenshot.png")
        speech_queue.put("Screenshot taken.")
    except Exception as e:
        speech_queue.put(f"Sorry, there was an error taking the screenshot: {str(e)}")

def math_operation(user_input):
    user_input = user_input.lower()
    result = None
    
    operations = ['add', 'plus', 'subtract', 'minus', 'multiply', 'times', 'divide', 'divided', 'sum', 'difference', 'product']
    parts = []
    
    for op in operations:
        if op in user_input:
            parts.extend(user_input.split(op))

    for part in parts:
        part = part.strip()

        if 'add' in part or 'plus' in part or 'sum' in part:
            try:
                numbers = [float(num) for num in part.split() if num.replace('.', '', 1).isdigit()]
                if len(numbers) == 2:
                    if result is None:
                        result = numbers[0] + numbers[1]
                    else:
                        result += numbers[1]
            except:
                pass
        
        elif 'subtract' in part or 'minus' in part or 'difference' in part:
            try:
                numbers = [float(num) for num in part.split() if num.replace('.', '', 1).isdigit()]
                if len(numbers) == 2:
                    if result is None:
                        result = numbers[0] - numbers[1]
                    else:
                        result -= numbers[1]
            except:
                pass

        elif 'multiply' in part or 'times' in part or 'product' in part:
            try:
                numbers = [float(num) for num in part.split() if num.replace('.', '', 1).isdigit()]
                if len(numbers) == 2:
                    if result is None:
                        result = numbers[0] * numbers[1]
                    else:
                        result *= numbers[1]
            except:
                pass

        elif 'divide' in part or 'divided' in part:
            try:
                numbers = [float(num) for num in part.split() if num.replace('.', '', 1).isdigit()]
                if len(numbers) == 2 and numbers[1] != 0:
                    if result is None:
                        result = numbers[0] / numbers[1]
                    else:
                        result /= numbers[1]
                elif numbers[1] == 0:
                    result = "Cannot divide by zero."
            except:
                pass

        try:
            expr = part.replace('plus', '+').replace('minus', '-').replace('times', '*').replace('divided', '/')
            result = eval(expr)
            result = round(result, 4)
        except:
            pass

    if result is None:
        return "Sorry, I couldn't understand the math operation."
    return round(result, 4)

def set_alarm(time_str, speech_queue, alarm_sound_path="alarm_sound.mp3"):
    try:
        alarm_time_obj = datetime.datetime.strptime(time_str, "%I:%M %p").time()
        now = datetime.datetime.now()
        alarm_time = datetime.datetime.combine(now.date(), alarm_time_obj)

        if alarm_time <= now:
            alarm_time += datetime.timedelta(days=1)

        time_until_alarm = (alarm_time - now).total_seconds()

        def trigger_alarm():
            pygame.mixer.init()
            pygame.mixer.music.load(alarm_sound_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                continue

        Timer(time_until_alarm, trigger_alarm).start()
        speech_queue.put(f"Alarm set for {time_str}.")

    except ValueError:
        speech_queue.put("Sorry, there was an error setting the alarm. Please provide a valid time format.")
        
def check_weather():
    url = "https://www.google.com/search?q=weather"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    try:
        temp = soup.find('div', class_="BNeawe iBp4i AP7Wnd").text
        condition = soup.find('div', class_="BNeawe tAd8D AP7Wnd").text
        weather_info = f"The current temperature is {temp} and the weather condition is {condition}."
        speech_queue.put(weather_info)
    except AttributeError:
        speech_queue.put("Sorry, I couldn't fetch the weather information at the moment.")
        


#* UTILITY FUNCTIONS

def play_beep():
    frequency = 1000  # Hz
    duration = 200    # milliseconds
    winsound.Beep(frequency, duration)

def wake_word_listener():
    q = Queue()
    samplerate = 16000

    def callback(indata, frames, time, status):
        if status:
            print("Stream status:", status)
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                            channels=1, callback=callback):
        print("Listening for wake word... (say 'Hey Jarvis')")

        rec = KaldiRecognizer(vosk_model, samplerate)
        while True:
            try:
                data = q.get(timeout=5)
            except:
                print("No audio received.")
                continue

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
            else:
                result = json.loads(rec.PartialResult())

            text = result.get("text", "")
            print("Detected text:", text)

            if wake_word in text.lower():
                print("Wake word detected!")
                return
            
def speak():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(str(text))
        engine.runAndWait()
        speech_queue.task_done()

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses']), tag
    else:
        return "I do not understand...", None

def perform_task(tag, query=None):
    if tag == "open_calculator":
        os.system("calc")
        
    elif tag == "open_browser":
        webbrowser.open("https://www.google.com")
        
    elif tag == "open_notepad":
        os.system("notepad")
        
    elif tag == "tell_time":
        now = datetime.datetime.now().strftime("%I:%M %p")
        speech_queue.put(f"The time is {now}")
        
    elif tag == "open_youtube":
        webbrowser.open("https://www.youtube.com")
        
    elif tag == "open_camera":
        os.system("start microsoft.windows.camera:")
        
    elif tag == "open_settings":
        os.system("start ms-settings:")
        
    elif  tag == "open_whatsapp":
        webbrowser.open("https://web.whatsapp.com")
        
    elif tag == "open_music":
        subprocess.Popen(["start", "wmplayer"], shell=True)
        
    elif tag == "open_instagram":
        webbrowser.open("https://www.instagram.com")
        
    elif tag == "open_facebook":
        webbrowser.open("https://www.facebook.com")
        
    elif tag == "open_maps":
        webbrowser.open("https://www.google.com/maps")
        
    elif tag == "open_email":
        webbrowser.open("https://www.gmail.com")
        
    elif tag == "open_twitter":
        webbrowser.open("https://www.twitter.com")

    elif tag == "open_reddit":
        webbrowser.open("https://www.reddit.com")

    elif tag == "open_amazon":
        webbrowser.open("https://www.amazon.in")

    elif tag == "open_linkedin":
        webbrowser.open("https://www.linkedin.com")
        
    elif tag == "open_gallery":
        os.startfile("C:\\Users\\User\\AppData\\Local\\Packages\\Microsoft.Windows.Photos_8wekyb3d8bbwe\\LocalState\\assets")
    
    elif tag == "open_file_explorer":
        os.system("explorer")
        
    elif tag == "open_cmd":
        os.system("start cmd")
        
    elif tag == "open_powershell":
        os.system("start powershell")
        
    elif tag == "open_word":
        os.system("start winword")
        
    elif tag == "open_powerpoint":
        os.system("start powerpnt")
        
    elif tag == "open_excel":
        os.system("start excel")
        
    elif tag == "open_access":
        os.system("start access")
        
    elif tag == "open_outlook":
        os.system("start outlook")
        
    elif tag == "open_teams":
        os.system("start teams")
        
    elif tag == "open_edge":
        os.system("start msedge")
        
    elif tag == "open_chrome":
        os.system("start chrome")
        
    elif tag == "open_brave":
        os.system("start brave")
        
    elif tag == "open_vscode":
        os.system("start code")
        
    elif tag == "open_task_manager":
        os.system("start taskmgr")
    elif tag == "open_calendar":
        webbrowser.open("https://calendar.google.com")

    elif tag == "open_drive":
        webbrowser.open("https://drive.google.com")

    elif tag == "open_onenote":
        os.system("start onenote")
        
    elif tag == "check_battery":
        battery = psutil.sensors_battery()
        speech_queue.put(f"The battery percentage is {battery.percent}%")
        
    elif tag == "check_date":
        date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        speech_queue.put(f"The date is {date}")
        
    elif tag == "check_weather":
        check_weather()
        
    elif tag == "set_alarm":
        time_str = extract_alarm_time(query)
        if time_str is not None:
            set_alarm(time_str,speech_queue)
        else:
            speech_queue.put("Sorry, I couldn't understand the time you mentioned. Please try again with a valid time format.")   
            
    elif tag == "search_google":
        query = extract_google_query(query)
        url = f"https://www.google.com/search?q={query}"
        webbrowser.open(url)
        
    elif tag == "math_operation":
        try:
            result = math_operation(query)
            speech_queue.put(result)
        except Exception as e:
            speech_queue.put(f"Sorry, there was an error with the math operation: {str(e)}")
            
    elif tag == "take_screenshot":
        take_screenshot()
        
    elif tag == "mute_volume":
        if is_muted():
            speech_queue.put("Volume is already muted.")
            return
        mute_volume()
        
    elif tag == "unmute_volume":
        if not is_muted():
            speech_queue.put("Volume is already unmuted.")
            return
        unmute_volume()
        
    elif tag == "volume_up":
        pyautogui.press("volumeup")
        
    elif tag == "volume_down":
        pyautogui.press("volumedown")
        
    elif tag == "increase_brightness":
        current = sbc.get_brightness()[0]
        sbc.set_brightness(min(current + 10, 100))
        
    elif tag == "decrease_brightness":
        current = sbc.get_brightness()[0]
        sbc.set_brightness(max(current - 10, 0))
    elif tag == "lock_screen":
        ctypes.windll.user32.LockWorkStation()

    elif tag == "shutdown":
        os.system("shutdown /s /t 1")

    elif tag == "restart":
        os.system("shutdown /r /t 1")

    elif tag == "logout_user":
        os.system("shutdown -l")
        
    else:
        speech_queue.put("I'm sorry, I don't understand that command.")

def listen(timeout=6):
    q = Queue()
    samplerate = 16000
    rec = KaldiRecognizer(vosk_model, samplerate)

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        print("Listening...")
        start_time = datetime.datetime.now()
        full_text = ""

        while (datetime.datetime.now() - start_time).total_seconds() < timeout:
            try:
                data = q.get(timeout=1)
            except:
                continue

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    full_text += " " + text
            else:
                partial = json.loads(rec.PartialResult())
                print("Partial:", partial.get("partial", ""))

        final = rec.FinalResult()
        final_text = json.loads(final).get("text", "")
        full_text += " " + final_text
        print(f"You said: {full_text.strip()}")
        return full_text.strip().lower()

def recognize_command_vosk(timeout=6):
    q = Queue()
    samplerate = 16000
    rec = KaldiRecognizer(vosk_model, samplerate)

    def callback(indata, frames, time, status):
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        print("Listening for command...")
        start_time = datetime.datetime.now()
        full_text = ""
        while (datetime.datetime.now() - start_time).total_seconds() < timeout:
            try:
                data = q.get(timeout=1)
            except:
                continue

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                full_text += " " + text
            else:
                partial = json.loads(rec.PartialResult())
                print("Partial:", partial.get("partial", ""))

        final = rec.FinalResult()
        final_text = json.loads(final).get("text", "")
        full_text += " " + final_text
        print("Final recognized command:", full_text.strip())
        return full_text.strip()
    
    
def recognize_command_vosk(timeout=6):
    q = Queue()
    samplerate = 16000
    rec = KaldiRecognizer(vosk_model, samplerate)

    def callback(indata, frames, time, status):
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        print("Listening for command...")
        start_time = datetime.datetime.now()
        full_text = ""

        while (datetime.datetime.now() - start_time).total_seconds() < timeout:
            try:
                data = q.get(timeout=1)
            except:
                continue

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    full_text += " " + text
            else:
                partial = json.loads(rec.PartialResult())
                print("Partial:", partial.get("partial", ""))

        final = rec.FinalResult()
        final_text = json.loads(final).get("text", "")
        full_text += " " + final_text
        print("Final recognized command:", full_text.strip())
        return full_text.strip()


def handle_input():
    global stop_listening
    while not stop_listening:
        wake_word_listener()
        play_beep()

        message = recognize_command_vosk(timeout=6)
        if not message:
            print("No command detected.")
            continue

        if "goodbye jarvis" in message:
            speech_queue.put("Goodbye!")
            stop_listening = True
            break

        response, tag = get_response(message)
        speech_queue.put(response)
        if tag:
            perform_task(tag, message)



Thread(target=speak, daemon=True).start()
handle_input()
