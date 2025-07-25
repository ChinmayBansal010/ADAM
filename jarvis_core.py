import pvporcupine
import sounddevice as sd
import asyncio
import edge_tts
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
join_event = Event()
nlp = spacy.load("en_core_web_md")

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
current_user_preferences = {}

class IntentReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        logging.info("IntentReloadHandler: File system event detected for %s", event.src_path)
        if event.src_path.endswith("intents.json"):
            logging.info("IntentReloadHandler: Detected change in intents.json. Reloading...")
            load_intents()

def load_intents():
    global intents
    logging.info("load_intents: Attempting to load intents.json...")
    try:
        with open("intents.json", "r", encoding="utf-8") as f:
            intents = json.load(f)
        logging.info("load_intents: Intents loaded: %d", len(intents["intents"]))
    except FileNotFoundError:
        logging.error("load_intents: intents.json not found. Make sure it's in the correct directory.")
        intents = {"intents": []}
    except json.JSONDecodeError as e:
        logging.error(f"load_intents: Error decoding intents.json: {e}")
        intents = {"intents": []}
    except Exception as e:
        logging.error(f"load_intents: Unexpected error loading intents: {e}")
        intents = {"intents": []}

def log_intent(tag):
    logging.info("log_intent: Logging intent usage for tag: %s", tag)
    intent_usage[tag] = intent_usage.get(tag, 0) + 1

async def speak_text(text):
    global current_user_preferences
    voice_id = current_user_preferences.get("voice", "en-IN-PrabhatNeural")
    logging.info(f"speak_text: Attempting to speak text: '{text[:50]}...' using voice_id: {voice_id}")
    try:
        audio_stream = BytesIO()
        communicate = edge_tts.Communicate(text, voice_id, rate="+15%")
        received_audio = False
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                audio_stream.write(chunk["data"])
                received_audio = True
        if not received_audio:
            logging.error("speak_text: TTS Error: No audio received from Edge TTS. Check network or voice ID.")
            return

        audio_stream.seek(0)
        audio = AudioSegment.from_file(audio_stream, format="mp3")

        if audio.frame_rate != 44100:
            logging.info(f"speak_text: Resampling audio from {audio.frame_rate}Hz to 44100Hz for playback compatibility.")
            audio = audio.set_frame_rate(44100)

        logging.info("speak_text: Playing audio via simpleaudio...")
        play_obj = sa.play_buffer(
            audio.raw_data,
            num_channels=audio.channels,
            bytes_per_sample=audio.sample_width,
            sample_rate=audio.frame_rate
        )
        play_obj.wait_done()
        logging.info("speak_text: Audio playback finished.")

    except Exception as e:
        logging.error(f"speak_text: TTS/Audio playback error: {e}", exc_info=True)
        
def speak_worker():
    logging.info("speak_worker: TTS worker thread started.")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        text = speech_queue.get()
        if text is None:
            logging.info("speak_worker: Received None, stopping TTS worker thread.")
            break
        try:
            loop.run_until_complete(speak_text(str(text)))
        except Exception as e:
            logging.error(f"speak_worker: TTS crash: {str(e)}")
        finally:
            speech_queue.task_done()
    logging.info("speak_worker: TTS worker thread finished.")

Thread(target=speak_worker, daemon=True).start()
logging.info("Main: TTS worker thread initiated.")

def notify_listening():
    response = random.choice(["Yes sir", "I'm listening", "Ready"])
    speech_queue.put(response)
    logging.info(f"notify_listening: Jarvis responded with: '{response}'")

def fuzzy_match(user_text, intents_data):
    best_tag = None
    best_score = 0
    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            score = fuzz.token_sort_ratio(user_text, pattern)
            if score > best_score:
                best_score = score
                best_tag = intent["tag"]
    logging.info(f"fuzzy_match: Best fuzzy match: tag='{best_tag}', score={best_score}")
    return best_tag, best_score

def spacy_match(user_text, intents_data):
    if user_text in intent_cache:
        cached_result = intent_cache[user_text]
        logging.info(f"spacy_match: Cache hit for '{user_text[:20]}...'. Result: {cached_result}")
        return cached_result
    user_doc = nlp(user_text)
    best_tag = None
    best_score = 0
    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            pattern_doc = nlp(pattern)
            try:
                similarity = user_doc.similarity(pattern_doc)
            except Exception as e:
                logging.warning(f"spacy_match: Error calculating similarity for '{user_text}' and '{pattern}': {e}")
                similarity = 0.0
            if similarity > best_score:
                best_score = similarity
                best_tag = intent["tag"]
    result = (best_tag, best_score) if best_score > 0.7 else (None, 0)
    intent_cache[user_text] = result
    logging.info(f"spacy_match: Best spaCy match: tag='{best_tag}', score={best_score}. Cached result.")
    return result

def get_response(msg, model, tags, all_words, device, intents_data):
    logging.info(f"get_response: Processing message: '{msg[:50]}...'")
    tag_scores = {}

    with ThreadPoolExecutor() as executor:
        fuzzy_future = executor.submit(fuzzy_match, msg, intents_data)
        spacy_future = executor.submit(spacy_match, msg, intents_data)

        fuzzy_tag, fuzzy_score = fuzzy_future.result()
        spacy_tag, spacy_score = spacy_future.result()
    
    logging.info(f"get_response: Fuzzy match result: {fuzzy_tag=} {fuzzy_score=}")
    logging.info(f"get_response: spaCy match result: {spacy_tag=} {spacy_score=}")

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words).reshape(1, -1)
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    nn_tag = tags[predicted.item()]
    nn_confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()
    logging.info(f"get_response: NN result: tag='{nn_tag}', confidence={nn_confidence:.4f}")

    if fuzzy_tag: tag_scores[fuzzy_tag] = tag_scores.get(fuzzy_tag, 0) + fuzzy_score * 0.3
    if spacy_tag: tag_scores[spacy_tag] = tag_scores.get(spacy_tag, 0) + spacy_score * 0.3
    if nn_confidence > 0.75: tag_scores[nn_tag] = tag_scores.get(nn_tag, 0) + nn_confidence * 0.4

    logging.info(f"get_response: Combined tag scores: {tag_scores}")

    if not tag_scores:
        logging.warning("get_response: No strong tag scores found. Returning default response.")
        return "I do not understand...", None

    final_tag = max(tag_scores, key=tag_scores.get)
    context["last_tag"] = final_tag
    log_intent(final_tag)
    logging.info(f"get_response: Final determined tag: '{final_tag}'")

    for intent in intents_data['intents']:
        if final_tag == intent['tag']:
            response = random.choice(intent['responses'])
            logging.info(f"get_response: Found response for tag '{final_tag}': '{response[:50]}...'")
            return response, final_tag

    logging.warning("get_response: Final tag did not match any intent. Returning default response.")
    return "I do not understand...", None

def wake_word_listener():
    global current_user_preferences
    if not current_user_preferences.get('wake_word_enabled', True):
        logging.info("wake_word_listener: Wake word disabled. Returning True immediately.")
        time.sleep(0.1)
        return True

    logging.info("wake_word_listener: Wake word enabled. Starting Porcupine listener.")
    porcupine = None
    q = Queue()
    try:
        porcupine = pvporcupine.create(access_key=PORCUPINE_ACCESS_KEY, keywords=["jarvis"])
        logging.info(f"wake_word_listener: Porcupine created. Sample rate: {porcupine.sample_rate}, Frame length: {porcupine.frame_length}")
        
        def callback(indata, frames, time_info, status):
            if status:
                logging.warning(f"wake_word_listener: Sounddevice callback status: {status}")
            q.put(indata.copy())

        with sd.InputStream(
            channels=1,
            samplerate=porcupine.sample_rate,
            dtype='int16',
            blocksize=porcupine.frame_length,
            callback=callback
        ):
            logging.info("wake_word_listener: Sounddevice InputStream opened for Porcupine. Waiting for wake word...")
            while True:
                try:
                    data = q.get(timeout=1)
                except Empty:
                    if stop_listening:
                        logging.info("wake_word_listener: stop_listening flag set. Exiting wake word loop.")
                        break
                    continue
                
                pcm = struct.unpack_from("h" * porcupine.frame_length, data)
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    logging.info("wake_word_listener: Wake word 'Jarvis' detected!")
                    break
        logging.info("wake_word_listener: Wake word loop finished.")
        return True
    except pvporcupine.PorcupineInvalidArgumentError as pie:
        logging.error(f"wake_word_listener: Porcupine Argument Error: {pie}. Check ACCESS_KEY or keyword list.")
        speech_queue.put("Wake word system configuration failed.")
        return False
    except Exception as e:
        logging.error(f"wake_word_listener: Unexpected Wake Word Error: {str(e)}")
        speech_queue.put("Wake word system failed.")
        return False
    finally:
        if porcupine:
            logging.info("wake_word_listener: Deleting Porcupine instance.")
            porcupine.delete()
        logging.info("wake_word_listener: Porcupine listener cleanup complete.")

def record_until_silence(max_duration=10):
    logging.info("record_until_silence: Starting audio recording process.")
    try:
        frames = []
        start_time = time.time()
        with sd.RawInputStream(samplerate=SAMPLERATE, blocksize=BLOCK_SIZE, dtype='int16', channels=1) as stream:
            logging.info(f"record_until_silence: Audio stream opened. Device: {stream.device}, Samplerate: {stream.samplerate}, Channels: {stream.channels}")
            while not stop_listening and (time.time() - start_time < max_duration):
                try:
                    block, overflowed = stream.read(BLOCK_SIZE)
                    if overflowed:
                        logging.warning(f"record_until_silence: Audio input stream overflowed: {overflowed} frames dropped. Consider increasing blocksize or reducing system load.")
                    
                    if block is None or len(block) == 0:
                        logging.warning("record_until_silence: Received empty audio block from stream. Skipping.")
                        continue

                    np_block = np.frombuffer(block, dtype=np.int16)
                    frames.append(np_block)

                except sd.PortAudioError as pa_err:
                    logging.error(f"record_until_silence: PortAudio Error during stream read: {pa_err}. Check mic or drivers.")
                    return None
                except Exception as e:
                    logging.error(f"record_until_silence: Unexpected error during stream read: {e}")
                    return None
            
            if stop_listening:
                logging.info("record_until_silence: Recording stopped due to stop_listening flag.")
                return None
            
        if not frames:
            logging.warning("record_until_silence: No audio frames were captured.")
            return None

        concatenated_audio = np.concatenate(frames)
        logging.info(f"record_until_silence: Successfully recorded {len(concatenated_audio) / SAMPLERATE:.2f} seconds of audio.")
        return concatenated_audio.astype(np.float32) / 32768.0

    except Exception as e:
        logging.error(f"record_until_silence: Recording setup failed: {e}")
        return None

def recognize_command(audio, whisper_model):
    logging.info("recognize_command: Starting audio transcription with FasterWhisper.")
    try:
        segments, info = whisper_model.transcribe(audio, beam_size=5, language="en")
        transcript = " ".join([seg.text for seg in segments]).strip().lower()
        logging.info(f"recognize_command: Transcription complete. Text: '{transcript[:50]}...' Language: {info.language}")
        return transcript
    except Exception as e:
        logging.error(f"recognize_command: Transcription Error: {str(e)}")
        return ""

def handle_input(model, tags, all_words, device, intents_data, whisper_model):
    global stop_listening
    logging.info("handle_input: Jarvis main input handler loop started.")
    if whisper_model is None:
        speech_queue.put("Speech recognition model failed to load.")
        logging.critical("handle_input: whisper_model is None. Cannot proceed with speech recognition.")
        time.sleep(3)
        return

    while not stop_listening:
        if not wake_word_listener():
            if stop_listening:
                logging.info("handle_input: stop_listening detected after wake_word_listener failed.")
                break
            logging.warning("handle_input: Wake word listener failed or wake word not detected. Looping.")
            continue

        if stop_listening:
            logging.info("handle_input: stop_listening detected after wake word detection.")
            break

        notify_listening()
        speech_queue.join() 

        audio = record_until_silence()
        
        if audio is None:
            if stop_listening:
                logging.info("handle_input: stop_listening detected after record_until_silence returned None.")
                break
            logging.warning("handle_input: record_until_silence returned None. Skipping transcription.")
            continue
        
        logging.info(f"handle_input: Audio captured successfully. Length: {len(audio)} samples. Min: {np.min(audio):.4f}, Max: {np.max(audio):.4f}")

        if stop_listening:
            logging.info("handle_input: Stop listening flag set after audio capture. Exiting loop.")
            break

        transcript = recognize_command(audio, whisper_model)
        logging.info(f"handle_input: Transcript received: '{transcript}'")

        if not transcript:
            if stop_listening:
                logging.info("handle_input: stop_listening detected after no transcript.")
                break
            logging.warning("handle_input: No transcript received. Skipping intent processing.")
            continue
        
        logging.info(f"handle_input: You said: {transcript}")
        
        response, tag = get_response(transcript, model, tags, all_words, device, intents_data)
        logging.info(f"handle_input: Response: '{response}', Tag: '{tag}'")
        
        speech_queue.put(response)
        
        if tag:
            try:
                perform_task(tag, transcript, speech_queue)
                logging.info(f"handle_input: Task '{tag}' performed successfully.")
            except Exception as e:
                logging.error(f"handle_input: Task execution failed for tag '{tag}': {str(e)}")

        time.sleep(1)

    logging.info("handle_input: Jarvis main input handler loop ended.")

def load_all_models_and_intents():
    logging.info("load_all_models_and_intents: Starting to load all core components.")
    model, tags, all_words, device, intents, whisper_model = None, None, None, None, None, None
    try:
        with open('intents.json', 'r', encoding="utf-8") as f:
            intents = json.load(f)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"load_all_models_and_intents: Using device: {device}")
        
        data = torch.load('model.pth', map_location=device)
        
        model = NeuralNet(data['input_size'], data['hidden_size'], data['output_size']).to(device)
        model.load_state_dict(data['model_state'])
        model.eval()
        logging.info("load_all_models_and_intents: NeuralNet model loaded successfully.")
        
        tags = data['tags']
        all_words = data['all_words']
        
        whisper_model = WhisperModel("small", compute_type="int8", device="cpu")
        logging.info("load_all_models_and_intents: FasterWhisper model loaded successfully.")
        
        logging.info("load_all_models_and_intents: All core components loaded successfully.")
        return model, tags, all_words, device, intents, whisper_model
    except FileNotFoundError as fnfe:
        logging.critical(f"load_all_models_and_intents: Required file not found: {fnfe}")
    except json.JSONDecodeError as jde:
        logging.critical(f"load_all_models_and_intents: Error decoding JSON: {jde}")
    except Exception as e:
        logging.critical(f"load_all_models_and_intents: Unexpected Model Load Error: {str(e)}")
    return None, None, None, None, None, None

def start_jarvis_listening(user_info, model, tags, all_words, device, intents_data, whisper_model):
    global stop_listening, current_user_preferences
    logging.info("start_jarvis_listening: Initiating Jarvis listening mode.")
    stop_listening = False
    current_user_preferences = user_info.get("preferences", {})
    logging.info(f"start_jarvis_listening: User preferences loaded: {current_user_preferences}")

    jarvis_thread = Thread(target=handle_input, args=(model, tags, all_words, device, intents_data, whisper_model))
    jarvis_thread.daemon = True
    jarvis_thread.start()
    logging.info("start_jarvis_listening: Jarvis input handler thread started.")

def stop_jarvis_listening():
    global stop_listening
    logging.info("stop_jarvis_listening: Setting stop_listening flag to True.")
    stop_listening = True
    logging.info("stop_jarvis_listening: Clearing speech queue.")
    with speech_queue.mutex:
        speech_queue.queue.clear()
    speech_queue.put(None)
    logging.info("stop_jarvis_listening: Jarvis listening stopped.")

if __name__ == "__main__":
    logging.info("Main: Jarvis application starting in standalone mode.")
    model, tags, all_words, device, intents, whisper_model = load_all_models_and_intents()
    if None in [model, tags, all_words, device, intents, whisper_model]:
        logging.critical("Main: Jarvis core components failed to load. Exiting application.")
    else:
        logging.info("Main: Jarvis core components loaded successfully. Starting listening loop.")
        temp_user_info = {"displayName": "Guest", "preferences": {"voice": "en-IN-PrabhatNeural", "wake_word_enabled": True}}
        start_jarvis_listening(temp_user_info, model, tags, all_words, device, intents, whisper_model)
        try:
            logging.info("Main: Entering main application loop. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Main: KeyboardInterrupt detected. Stopping Jarvis listening.")
            stop_jarvis_listening()
            logging.info("Main: Exiting Jarvis standalone gracefully.")
        except Exception as e:
            logging.error(f"Main: Unexpected Error: {str(e)}")
        finally:
            logging.info("Main: Ensuring Jarvis listening is stopped in finally block.")
            stop_jarvis_listening()