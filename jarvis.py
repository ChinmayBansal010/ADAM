import pvporcupine
import sounddevice as sd
import asyncio
import edge_tts
import webrtcvad
# from windows_toasts import Toast, WindowsToaster
from pydub import AudioSegment
import simpleaudio as sa
import torch
import numpy as np
import json
import struct
import time
import random
import logging
from io import BytesIO
from dotenv import load_dotenv
import os
from threading import Thread, Event
from queue import Queue,Empty
from faster_whisper import WhisperModel
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from task import perform_task
import nltk

# === Set nltk data path ===
try:
    nltk.download('punkt_tab')
except Exception:
    pass
nltk.data.path.append("C:/Users/chinm/AppData/Roaming/nltk_data")
# === Load environment ===
load_dotenv()
ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

# === Constants ===
SAMPLERATE = 16000
BLOCK_DURATION = 30
BLOCK_SIZE = int(SAMPLERATE * BLOCK_DURATION / 1000)

# === Global variables ===
speech_queue = Queue()
stop_listening = False
vad = webrtcvad.Vad(2)
join_event = Event()

# === Toast notifier ===
# try:
#     toaster = WindowsToaster("Jarvis Assistant")
# except Exception:
#     toaster = None

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jarvis.log"),
        logging.StreamHandler()
    ]
)

# === Background speech thread ===
async def speak_text(text):
    try:
        audio_stream = BytesIO()
        communicate = edge_tts.Communicate(text, "en-IN-PrabhatNeural", rate="+15%")

        received_audio = False
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                audio_stream.write(chunk["data"])
                received_audio = True

        if not received_audio:
            logging.error("TTS Error: No audio received.")
            notify("TTS Error", "No audio received. Check network or voice ID.")
            return

        audio_stream.seek(0)
        audio = AudioSegment.from_file(audio_stream, format="mp3")

        # === In-memory playback with simpleaudio ===
        raw_data = audio.raw_data
        play_obj = sa.play_buffer(
            raw_data,
            num_channels=audio.channels,
            bytes_per_sample=audio.sample_width,
            sample_rate=audio.frame_rate
        )
        play_obj.wait_done()

    except Exception as e:
        notify("TTS Error", str(e))
        logging.error(f"TTS Error: {str(e)}")
def speak_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            loop.run_until_complete(speak_text(str(text)))
        except Exception as e:
            notify("TTS crash", str(e))
            logging.error(f"TTS crash: {str(e)}")
        finally:
            speech_queue.task_done()

Thread(target=speak_worker, daemon=True).start()

def notify_listening():
    response = random.choice(["Yes sir", "I'm listening", "Ready"])
    speech_queue.put(response)
    notify("Jarvis", response)

def notify(title="Jarvis", message="Message"):
    logging.info(f"{title}: {message}")
        
def get_response(msg, model, tags, all_words, device, intents):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words).reshape(1, -1)
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    if probs[0][predicted.item()] > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses']), tag
    return "I do not understand...", None

def wake_word_listener():
    try:
        porcupine = pvporcupine.create(access_key=ACCESS_KEY, keywords=["jarvis"])
        q = Queue()

        def callback(indata, frames, time_info, status):
            q.put(indata.copy())

        with sd.InputStream(
            channels=1,
            samplerate=porcupine.sample_rate,
            dtype='int16',
            blocksize=porcupine.frame_length,
            callback=callback
        ):
            while True:
                try:
                    data = q.get(timeout=1)
                except Empty:
                    if stop_listening:
                        break
                pcm = struct.unpack_from("h" * porcupine.frame_length, data)
                if porcupine.process(pcm) >= 0:
                    break
        porcupine.delete()
    except Exception as e:
        notify("Wake Word Error", str(e))
        logging.error(f"Wake Word Error: {str(e)}")
        speech_queue.put("Wake word system failed.")

def record_until_silence(max_duration=10, silence_duration=1.2):
    try:
        frames = []
        start_time = time.time()
        silence_start = None
        logging.info("Listening for command...")

        with sd.RawInputStream(samplerate=SAMPLERATE, blocksize=BLOCK_SIZE, dtype='int16', channels=1) as stream:
            while True:
                block = stream.read(BLOCK_SIZE)[0]
                if vad.is_speech(block, SAMPLERATE):
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
        return np.concatenate(frames).astype(np.float32) / 32768.0
    except Exception as e:
        notify("Recording failed: ",str(e))
        logging.error(f"Recording failed: {str(e)}")
        return None

def recognize_command(audio, whisper_model):
    try:
        segments, _ = whisper_model.transcribe(audio, beam_size=5, language="en")
        return " ".join([seg.text for seg in segments]).strip().lower()
    except Exception as e:
        notify("Transcription Error", str(e))
        return ""

def handle_input(model, tags, all_words, device, intents, whisper_model):
    global stop_listening
    if whisper_model is None:
        speech_queue.put("Speech recognition model failed to load.")
        time.sleep(3)
        return

    while not stop_listening:
        wake_word_listener()
        notify_listening()
        
        # Wait for the "I'm listening" TTS to finish before recording
        speech_queue.join()

        audio = record_until_silence()
        if audio is None:
            notify("Error: "," No speech detected.")
            continue

        transcript = recognize_command(audio, whisper_model)
        if not transcript:
            notify("Error: ","Could not understand.")
            continue

        logging.info(f"You said: {transcript}")

        if "goodbye jarvis" in transcript:
            speech_queue.put("Goodbye!")
            stop_listening = True
            break

        response, tag = get_response(transcript, model, tags, all_words, device, intents)
        speech_queue.put(response)
        if tag:
            try:
                perform_task(tag, transcript, speech_queue)
            except Exception as e:
                notify("Task execution failed: ", str(e))
                logging.error(f"Task execution failed: {str(e)}")

def join_with_timeout(queue, timeout=3):
    def wait_and_set():
        queue.join()
        join_event.set()
    t = Thread(target=wait_and_set)
    t.start()
    t.join(timeout)
    if not join_event.is_set():
        notify("TTS did not finish in time", "Exiting forcefully.")
        logging.warning("TTS did not finish in time, exiting forcefully.")
        
def load_all():
    try:
        with open('intents.json', 'r') as f:
            intents = json.load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.load('model.pth', map_location=device)

        model = NeuralNet(data['input_size'], data['hidden_size'], data['output_size']).to(device)
        model.load_state_dict(data['model_state'])
        model.eval()

        whisper_model = WhisperModel("small", compute_type="int8", device="cpu")

        return model, data['tags'], data['all_words'], device, intents, whisper_model

    except Exception as e:
        notify("Model Load Error", str(e))
        logging.error(f"Model Load Error: {str(e)}")
        return None, None, None, None, None, None

if __name__ == "__main__":
    try:
        model, tags, all_words, device, intents, whisper_model = load_all()
        if None in [model, tags, all_words, device, intents, whisper_model]:
            speech_queue.put("Startup failed.")
        else:
            handle_input(model, tags, all_words, device, intents, whisper_model)
    except KeyboardInterrupt:
        notify("Bye", "Exiting Jarvis.")
        logging.info("Exiting Jarvis.")
    except Exception as e:
        notify("Unexpected Error", str(e))
        logging.error(f"Unexpected Error: {str(e)}")
    finally:
        with speech_queue.mutex:
            speech_queue.queue.clear()
        speech_queue.put(None)