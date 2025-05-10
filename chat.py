import random
import pyttsx3
import speech_recognition as sr
import torch
import json
import os
import re
import webbrowser
from bs4 import BeautifulSoup
import requests
import datetime
import time
from word2number import w2n
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from tkinter import *
from threading import Thread, Timer
import subprocess
import pygame
import psutil
from queue import Queue

engine = pyttsx3.init()
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = False
recognizer.energy_threshold = 100

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

def math_operation(user_input):
    user_input = user_input.lower()
    result = None

    # Split the input into multiple parts based on common operation words
    operations = ['add', 'plus', 'subtract', 'minus', 'multiply', 'times', 'divide', 'divided', 'sum', 'difference', 'product']
    parts = []
    for op in operations:
        if op in user_input:
            parts.extend(user_input.split(op))

    # Now process each part separately
    for part in parts:
        part = part.strip()

        # Handle "add 20 and 30" or "the sum of 20 and 30"
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
        
        # Handle "subtract 40 and 50" or "difference between 40 and 50"
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

        # Handle "multiply 5 and 7" or "product of 5 and 7"
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

        # Handle "divide 40 by 5" or "divide 20 and 5"
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

        # Fallback for more complex expressions
        try:
            # Try parsing mathematical expressions like "20 + 30"
            expr = part.replace('plus', '+').replace('minus', '-').replace('times', '*').replace('divided', '/')
            result = eval(expr)
            result = round(result, 4)
        except:
            pass

    if result is None:
        return "Sorry, I couldn't understand the math operation."
    return round(result, 4)


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
# Function to set the alarm
def set_alarm(time_str):
    try:
        alarm_time_obj = datetime.datetime.strptime(time_str, "%I:%M %p").time()
        now = datetime.datetime.now() 
        alarm_time = datetime.datetime.combine(now.date(), alarm_time_obj)

        if alarm_time <= now:
            alarm_time += datetime.timedelta(days=1)

        time_until_alarm = (alarm_time - now).total_seconds()  

        def trigger_alarm():
            pygame.mixer.init()
            pygame.mixer.music.load("alarm_sound.mp3")  
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                continue
    
        Timer(time_until_alarm, trigger_alarm).start()

    except ValueError:
        speech_queue.put("Sorry, there was an error setting the alarm. Please provide a valid time format.")
        
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
    elif tag == "check_battery":
        battery = psutil.sensors_battery()
        speech_queue.put(f"The battery percentage is {battery.percent}%")
    elif tag == "check_weather":
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
    elif tag == "set_alarm":
        time_str = extract_alarm_time(query) 
        print(time_str)
        if time_str is not None:
            set_alarm(time_str)
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

def listen():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        status_var.set("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=4)
    try:
        command = recognizer.recognize_google(audio)
        print(f"You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

def handle_input():
    global stop_listening
    while True:
        if stop_listening:
            break
        message = listen()

        if message == '':
            continue
        if message == 'quit':
            break
        response, tag = get_response(message)
        speech_queue.put(response)
        if tag:
            perform_task(tag, message)

def start_stop():
    global stop_listening
    stop_listening = not stop_listening
    if not stop_listening:
        start_btn.config(text="Stop Listening", bg="#ff6666")
        status_var.set("Listening...")
        animate_status()
        Thread(target=handle_input, daemon=True).start()
    else:
        status_var.set("Stopped.")
        start_btn.config(text="Start Listening", bg="#4CAF50")


def animate_status():
    if not stop_listening:
        current = status_label.cget("fg")
        new_color = "#0066cc" if current == "#003366" else "#003366"
        status_label.config(fg=new_color)
        root.after(500, animate_status)

# Enhanced GUI setup
root = Tk()
root.title("Voice Assistant Chatbot")
root.geometry("450x300")
root.configure(bg="#e6f2ff")

label = Label(root, text="Voice Assistant", font=("Helvetica", 24, "bold"), bg="#e6f2ff", fg="#003366")
label.pack(pady=20)

status_var = StringVar()
status_var.set("Click Start to begin...")

status_label = Label(root, textvariable=status_var, font=("Helvetica", 14), bg="#e6f2ff", fg="#003366")
status_label.pack(pady=10)

start_btn = Button(root, text="Start Listening", font=("Helvetica", 14, "bold"),
                   command=start_stop, bg="#4CAF50", fg="white", padx=20, pady=10)
start_btn.pack(pady=30)

Thread(target=speak, daemon=True).start()
root.mainloop()
