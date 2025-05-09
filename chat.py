import random
import pyttsx3
import speech_recognition as sr
import torch
import json
import os
import webbrowser
import datetime
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from tkinter import *
from threading import Thread
from queue import Queue

engine = pyttsx3.init()
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = False
recognizer.energy_threshold = 200

speech_queue = Queue()

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'model.pth'
data = torch.load(FILE)

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
        engine.say(text)
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

def extract_google_query(message):
    keywords = ["search google for", "google search for" ,"search for", "search google" , "google search", "google for", "google", "find", "search"]
    msg = message.lower()
    for kw in keywords:
        if kw in msg:
            return msg.split(kw)[-1].strip()
    return msg

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
    elif tag == "search_google":
        query = extract_google_query(query)
        url = f"https://www.google.com/search?q={query}"
        webbrowser.open(url)

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
