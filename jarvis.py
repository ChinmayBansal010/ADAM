import pvporcupine
import sounddevice as sd
import asyncio
import edge_tts
import webrtcvad
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
from queue import Queue, Empty
from faster_whisper import WhisperModel
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from task import perform_task
import spacy
from rapidfuzz import fuzz
from concurrent.futures import ThreadPoolExecutor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

load_dotenv()
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

SAMPLERATE = 16000
BLOCK_DURATION = 30
BLOCK_SIZE = int(SAMPLERATE * BLOCK_DURATION / 1000)

speech_queue = Queue()
stop_listening = False
vad = webrtcvad.Vad(2)
join_event = Event()
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jarvis.log"),
        logging.StreamHandler()
    ]
)

intent_cache = {}
intent_usage = {}
context = {"last_tag": None}

class IntentReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("intents.json"):
            logging.info("Detected change in intents.json. Reloading...")
            load_intents()

def load_intents():
    global intents
    with open("intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)
    logging.info("Intents loaded: %d", len(intents["intents"]))

def log_intent(tag):
    intent_usage[tag] = intent_usage.get(tag, 0) + 1
    logging.info(f"Intent used: {tag} ({intent_usage[tag]} times)")

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

def notify(title="Jarvis", message="Message"):
    logging.info(f"{title}: {message}")

def notify_listening():
    response = random.choice(["Yes sir", "I'm listening", "Ready"])
    speech_queue.put(response)
    notify("Jarvis", response)

def fuzzy_match(user_text, intents):
    best_tag = None
    best_score = 0
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            score = fuzz.token_sort_ratio(user_text, pattern)
            if score > best_score:
                best_score = score
                best_tag = intent["tag"]
    return best_tag, best_score

def spacy_match(user_text, intents):
    if user_text in intent_cache:
        return intent_cache[user_text]
    user_doc = nlp(user_text)
    best_tag = None
    best_score = 0
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_doc = nlp(pattern)
            similarity = user_doc.similarity(pattern_doc)
            if similarity > best_score:
                best_score = similarity
                best_tag = intent["tag"]
    result = (best_tag, best_score) if best_score > 0.7 else (None, 0)
    intent_cache[user_text] = result
    return result

def get_response(msg, model, tags, all_words, device, intents):
    tag_scores = {}

    with ThreadPoolExecutor() as executor:
        fuzzy_future = executor.submit(fuzzy_match, msg, intents)
        spacy_future = executor.submit(spacy_match, msg, intents)

        fuzzy_tag, fuzzy_score = fuzzy_future.result()
        spacy_tag, spacy_score = spacy_future.result()

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words).reshape(1, -1)
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    nn_tag = tags[predicted.item()]
    nn_confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()

    if fuzzy_tag: tag_scores[fuzzy_tag] = tag_scores.get(fuzzy_tag, 0) + fuzzy_score * 0.3
    if spacy_tag: tag_scores[spacy_tag] = tag_scores.get(spacy_tag, 0) + spacy_score * 0.3
    if nn_confidence > 0.75: tag_scores[nn_tag] = tag_scores.get(nn_tag, 0) + nn_confidence * 0.4

    if not tag_scores:
        return "I do not understand...", None

    final_tag = max(tag_scores, key=tag_scores.get)
    context["last_tag"] = final_tag
    log_intent(final_tag)

    for intent in intents['intents']:
        if final_tag == intent['tag']:
            return random.choice(intent['responses']), final_tag

    return "I do not understand...", None

def wake_word_listener():
    try:
        porcupine = pvporcupine.create(access_key=PORCUPINE_ACCESS_KEY, keywords=["jarvis"])
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
        notify("Recording failed", str(e))
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
        speech_queue.join()
        audio = record_until_silence()
        if audio is None:
            notify("Error", "No speech detected.")
            continue
        transcript = recognize_command(audio, whisper_model)
        if not transcript:
            notify("Error", "Could not understand.")
            continue
        logging.info(f"You said: {transcript}")
        response, tag = get_response(transcript, model, tags, all_words, device, intents)
        speech_queue.put(response)
        if tag:
            try:
                perform_task(tag, transcript, speech_queue)
            except Exception as e:
                notify("Task execution failed", str(e))
                logging.error(f"Task execution failed: {str(e)}")

def load_all():
    try:
        with open('intents.json', 'r') as f:
            intents = json.load(f)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.load('model.pth', map_location=device)
        model = NeuralNet(data['input_size'], data['hidden_size'], data['output_size']).to(device)
        model.load_state_dict(data['model_state'])
        model.eval()
        whisper_model = WhisperModel("small", compute_type="int8", device="cuda")
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
