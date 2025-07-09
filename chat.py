import os
import random
import queue
import struct
import time
import numpy as np
import sounddevice as sd
import pyttsx3
import pvporcupine
import webrtcvad
import torch
import json
from faster_whisper import WhisperModel
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from task import perform_task
from threading import Thread
from queue import Queue
import winsound

#* CONFIGS & MODELS

samplerate = 16000
block_duration = 30  # ms
block_size = int(samplerate * block_duration / 1000)
vad = webrtcvad.Vad(2)

speech_queue = Queue()
stop_listening = False

whisper_model = WhisperModel("small", compute_type="int8")

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

engine = pyttsx3.init()
engine.setProperty('rate', 150)

#* UTILITIES

def speak():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(str(text))
        engine.runAndWait()
        speech_queue.task_done()

def play_beep():
    winsound.Beep(1000, 200)

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
    return "I do not understand...", None

#* WAKE WORD

def wake_word_listener():
    porcupine = pvporcupine.create(keywords=["jarvis"])
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    with sd.InputStream(channels=1, samplerate=porcupine.sample_rate, dtype='int16',
                        blocksize=porcupine.frame_length, callback=callback):
        print("🎙️ Listening for wake word...")
        while True:
            data = q.get()
            pcm = struct.unpack_from("h" * porcupine.frame_length, data)
            result = porcupine.process(pcm)
            if result >= 0:
                print("✅ Wake word detected")
                break
    porcupine.delete()

#* SPEECH RECORDING UNTIL SILENCE

def record_until_silence(max_duration=10, silence_duration=1.2):
    frames = []
    start_time = time.time()
    silence_start = None
    print("🎤 Listening for command...")

    with sd.RawInputStream(samplerate=samplerate, blocksize=block_size, dtype='int16', channels=1) as stream:
        while True:
            block = stream.read(block_size)[0]
            if vad.is_speech(block, samplerate):
                frames.append(np.frombuffer(block, dtype=np.int16))
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_duration:
                    break
            if time.time() - start_time > max_duration:
                break

    if not frames:
        return None
    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    return audio

#* WHISPER TRANSCRIPTION

def recognize_command(audio):
    segments, _ = whisper_model.transcribe(audio, beam_size=5)
    return " ".join([seg.text for seg in segments]).strip().lower()

#* MAIN LOOP

def handle_input():
    global stop_listening
    while not stop_listening:
        wake_word_listener()
        play_beep()
        audio = record_until_silence()
        if audio is None:
            print("❌ No speech detected.")
            continue
        transcript = recognize_command(audio)
        if not transcript:
            print("❌ No valid command recognized.")
            continue
        print(f"🗣️ You said: {transcript}")
        if "goodbye jarvis" in transcript:
            speech_queue.put("Goodbye!")
            stop_listening = True
            break
        response, tag = get_response(transcript)
        speech_queue.put(response)
        if tag:
            perform_task(tag, transcript, speech_queue)

#* START

Thread(target=speak, daemon=True).start()
handle_input()
