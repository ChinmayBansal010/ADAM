import os
import re
import webbrowser
import datetime
import subprocess
import operator
import ctypes
import psutil
import pyautogui
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from threading import Timer
from pydub import AudioSegment
from pydub.playback import play
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import screen_brightness_control as sbc
import google.generativeai as genai
import logging

# Configure logging for task.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jarvis.log"), # Use the same log file as core
        logging.StreamHandler()
    ]
)

load_dotenv()
GEMINI_ACCESS_KEY = os.getenv("GEMINI_ACCESS_KEY")

# Initialize Pycaw for Windows audio control
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
except Exception as e:
    logging.warning(f"Pycaw initialization failed (likely not on Windows): {e}")
    volume = None # Set to None if initialization fails

def mute_volume():
    if volume:
        volume.SetMute(1, None)
        logging.info("Volume muted.")
    else:
        logging.warning("Volume control not available (Pycaw not initialized).")

def unmute_volume():
    if volume:
        volume.SetMute(0, None)
        logging.info("Volume unmuted.")
    else:
        logging.warning("Volume control not available (Pycaw not initialized).")

def is_muted():
    if volume:
        return volume.GetMute() == 1
    return False # Assume not muted if control not available

def take_screenshot(speech_queue):
    try:
        screenshot_filename = "screenshot.png"
        pyautogui.screenshot().save(screenshot_filename)
        speech_queue.put("Screenshot taken.")
        logging.info(f"Screenshot saved to {screenshot_filename}")
    except Exception as e:
        speech_queue.put(f"Sorry, there was an error taking the screenshot: {str(e)}")
        logging.error(f"Failed to take screenshot: {e}", exc_info=True)

def _safe_eval(expr):
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        'x': operator.mul, # For "times"
    }
    
    # Attempt to parse basic "number operator number"
    # Allows for optional spaces around operators and negative numbers
    match = re.match(r"(-?\d+\.?\d*)\s*([\+\-\*\/x])\s*(-?\d+\.?\d*)", expr)
    
    if match:
        try:
            num1 = float(match.group(1))
            op_char = match.group(2)
            num2 = float(match.group(3))
            
            op_func = ops.get(op_char)
            if op_func:
                if op_char == '/' and num2 == 0:
                    logging.warning("Division by zero attempt.")
                    return "Cannot divide by zero."
                result = op_func(num1, num2)
                logging.info(f"Calculated: {num1} {op_char} {num2} = {result}")
                return result
        except ValueError as ve:
            logging.error(f"Error parsing numbers for calculation: {ve}", exc_info=True)
            return None
    logging.warning(f"Could not parse simple expression: '{expr}' for calculation.")
    return None

def math_operation(query):
    query = query.lower().replace('what is', '').replace('calculate', '').strip()
    query = query.replace('plus', '+').replace('add', '+')
    query = query.replace('minus', '-').replace('subtract', '-')
    query = query.replace('times', '*').replace('multiplied by', '*')
    query = query.replace('divided by', '/').replace('divide', '/')
    
    result = _safe_eval(query)
    if result is not None:
        # Use a more controlled rounding or format for speech
        if isinstance(result, float):
            return f"The result is {round(result, 2)}" # Round to 2 decimal places for speech
        return f"The result is {result}"
    return "Sorry, I couldn't perform that calculation."

def set_alarm(time_str, speech_queue, alarm_sound_path="alarm_sound.mp3"):
    try:
        # Make alarm sound path absolute
        absolute_alarm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), alarm_sound_path)
        
        alarm_time_obj = datetime.datetime.strptime(time_str, "%I:%M %p").time()
        now = datetime.datetime.now()
        alarm_time = datetime.datetime.combine(now.date(), alarm_time_obj)
        if alarm_time <= now:
            alarm_time += datetime.timedelta(days=1) # Set for next day if time has passed
        time_until_alarm = (alarm_time - now).total_seconds()

        if time_until_alarm <= 0:
            speech_queue.put("The time you specified is in the past. Please try a future time.")
            logging.warning(f"Attempted to set alarm for past time: {time_str}")
            return

        def trigger_alarm():
            try:
                logging.info(f"Triggering alarm using sound file: {absolute_alarm_path}")
                play(AudioSegment.from_file(absolute_alarm_path))
                speech_queue.put("Alarm!") # Notify user via speech
            except FileNotFoundError:
                speech_queue.put(f"Alarm sound file not found at {alarm_sound_path}. Please check the file.")
                logging.error(f"Alarm sound file not found: {absolute_alarm_path}")
            except Exception as e:
                speech_queue.put(f"Failed to play alarm: {str(e)}")
                logging.error(f"Error playing alarm sound: {e}", exc_info=True)

        Timer(time_until_alarm, trigger_alarm).start()
        speech_queue.put(f"Alarm set for {time_str}.")
        logging.info(f"Alarm set for {time_str} ({time_until_alarm:.0f} seconds from now).")
    except ValueError:
        speech_queue.put("Sorry, I couldn't understand the alarm time. Please specify a time like '5:30 PM'.")
        logging.error(f"Invalid time format for alarm: {time_str}", exc_info=True)
    except Exception as e:
        speech_queue.put(f"Sorry, there was an error setting the alarm: {str(e)}")
        logging.error(f"Unexpected error setting alarm: {e}", exc_info=True)

def check_weather(speech_queue=None):
    # Using a simple web scrape of Google Weather, which can be fragile.
    # Consider using a weather API (e.g., OpenWeatherMap, WeatherAPI.com) for robustness.
    # Example: https://openweathermap.org/api
    
    # Current location is Delhi, Delhi, India as per current time.
    location = "Delhi, India" # Hardcoded for now based on context

    try:
        url = f"https://www.google.com/search?q=weather+in+{location.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        temp_element = soup.find('div', class_="BNeawe iBp4i AP7Wnd")
        condition_element = soup.find('div', class_="BNeawe tAd8D AP7Wnd")
        
        if temp_element and condition_element:
            temp = temp_element.text
            condition = condition_element.text
            speech_queue.put(f"The current temperature in {location} is {temp} and the weather condition is {condition}.")
            logging.info(f"Fetched weather for {location}: {temp}, {condition}")
        else:
            speech_queue.put("Sorry, I couldn't find the weather information for that location.")
            logging.warning(f"Could not parse weather elements from Google for {location}. HTML structure may have changed.")
    except requests.exceptions.RequestException as req_err:
        speech_queue.put(f"Sorry, I couldn't connect to the internet to fetch weather: {req_err}")
        logging.error(f"Network error fetching weather for {location}: {req_err}", exc_info=True)
    except Exception as e:
        speech_queue.put(f"Sorry, there was an error fetching the weather information: {str(e)}")
        logging.error(f"Unexpected error fetching weather for {location}: {e}", exc_info=True)

def extract_alarm_time(message):
    # Improved regex to handle "at 5:30 PM" or "5 30 PM" or "set alarm for 6 AM"
    match = re.search(r"(\d{1,2})\s*(?:[:.]\s*(\d{2}))?\s*(a\.m\.|p\.m\.|am|pm)?", message, re.IGNORECASE)
    if match:
        hour_str = match.group(1)
        minute_str = match.group(2) if match.group(2) else "00" # Default minutes to 00 if not specified
        ampm = match.group(3)

        try:
            hour = int(hour_str)
            minute = int(minute_str)

            if ampm:
                if "p" in ampm.lower() and hour != 12:
                    hour += 12
                elif "a" in ampm.lower() and hour == 12: # 12 AM is 00:00
                    hour = 0
            
            # Basic validation for hour/minute
            if not (0 <= hour <= 23) or not (0 <= minute <= 59):
                logging.warning(f"Invalid time components extracted: H={hour}, M={minute}")
                return None
            
            # Format for datetime.strptime
            return f"{hour:02d}:{minute:02d} {'AM' if ampm and 'a' in ampm.lower() else 'PM' if ampm else ''}".strip()
        except ValueError:
            logging.error(f"Could not convert extracted time parts to int: {hour_str}, {minute_str}")
            return None
    logging.info(f"No alarm time found in message: '{message}'")
    return None

def extract_google_query(message):
    keywords = [
        r"search google for", r"google search for", r"search for", r"search google", r"google search", r"google for",
        r"can you google", r"could you search", r"please google", r"find information on", r"look up", r"search", r"google",
        r"find", r"can you look up", r"could you look up", r"i need info on", r"i want to know about", r"tell me about",
        r"can you find me", r"what is", r"who is", r"what are", r"how to", r"how do i"
    ]
    msg_lower = message.lower()
    
    for kw_pattern in keywords:
        # Use word boundaries (\b) to match whole words and be more precise
        match = re.search(rf"\b{kw_pattern}\b\s+(.*)", msg_lower)
        if match:
            query = match.group(1).strip()
            # Remove trailing punctuation that might be part of speech recognition
            query = re.sub(r'[.!?]$', '', query)
            logging.info(f"Extracted Google query: '{query}' from '{message}'")
            return query
    
    # If no specific keyword is found, assume the whole message is the query (as a fallback)
    logging.info(f"No specific Google search keyword found. Assuming full message as query: '{message}'")
    return message.strip()


def diagnose_symptoms(symptoms, speech_queue):
    if not GEMINI_ACCESS_KEY:
        speech_queue.put("I cannot provide medical advice. Gemini API key is not configured.")
        logging.error("Diagnose Symptoms: GEMINI_ACCESS_KEY is not set.")
        return

    try:
        genai.configure(api_key=GEMINI_ACCESS_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            f"You are a trusted virtual medical assistant. A user reports the following symptoms: {symptoms}.\n"
            "Respond in a short and clear format. Use concise bullet points for each of the following:\n"
            "- Possible disease(s) (max 2)\n"
            "- 1-2 Precautions\n"
            "- 1-2 Over-the-counter treatment options\n"
            "- When to consult a doctor (one sentence)\n"
            "- A one-line disclaimer that this is not a substitute for medical advice."
        )
        logging.info(f"Sending prompt to Gemini: '{prompt[:100]}...'")
        response = model.generate_content(prompt)
        clean_text = clean_response(response.text)
        speech_queue.put(clean_text)
        logging.info(f"Received and processed Gemini diagnosis for '{symptoms[:50]}'")

    except Exception as e:
        speech_queue.put(f"Sorry, I failed to diagnose symptoms due to an internal error: {str(e)}")
        logging.error(f"Error during Gemini diagnosis for '{symptoms}': {e}", exc_info=True)
    

def clean_response(text):
    text = re.sub(r"[#*`~•→⇒➤▶️▪️✅➔➥➽➧➠➤➢➣➞➟➡️➩➫➬➭➯➲➳➵➸➺➻➼➽➾]", "", text)
    text = re.sub(r"^\s*[-–—]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text) # Reduce excessive newlines
    text = re.sub(r"\s+([.,!?;:])", r"\1", text) # Remove space before punctuation
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"[^\x00-\x7F]+", " ", text) # Remove non-ASCII characters

    return text.strip()

def run_windows_command(command, speech_queue, success_msg="Command executed.", error_prefix="Failed to execute command"):
    try:
        subprocess.Popen(command, shell=True) # Use shell=True for 'start' command
        speech_queue.put(success_msg)
        logging.info(f"Windows command '{' '.join(command) if isinstance(command, list) else command}' executed.")
    except FileNotFoundError:
        speech_queue.put(f"{error_prefix}: Application not found or not in PATH.")
        logging.error(f"FileNotFoundError for command: {command}", exc_info=True)
    except Exception as e:
        speech_queue.put(f"{error_prefix}: {str(e)}")
        logging.error(f"Error executing command: {command}. Error: {e}", exc_info=True)

def perform_task(tag, query=None, speech_queue=None):
    logging.info(f"perform_task: Tag received: {tag}, Query: '{query}'")

    try:
        if tag == "open_calculator":
            run_windows_command("calc", speech_queue, "Opening calculator.", "Failed to open calculator")
        elif tag == "open_browser":
            webbrowser.open("https://www.google.com")
            speech_queue.put("Opening web browser.")
        elif tag == "open_notepad":
            run_windows_command("notepad", speech_queue, "Opening notepad.", "Failed to open notepad")
        elif tag == "tell_time":
            speech_queue.put(datetime.datetime.now().strftime("The time is %I:%M %p"))
        elif tag == "open_youtube":
            webbrowser.open("https://www.youtube.com") # Corrected URL
            speech_queue.put("Opening YouTube.")
        elif tag == "open_camera":
            run_windows_command(["start", "microsoft.windows.camera:"], speech_queue, "Opening camera.", "Failed to open camera")
        elif tag == "open_settings":
            run_windows_command(["start", "ms-settings:"], speech_queue, "Opening settings.", "Failed to open settings")
        elif tag == "open_whatsapp":
            webbrowser.open("https://web.whatsapp.com")
            speech_queue.put("Opening WhatsApp web.")
        elif tag == "open_music":
            run_windows_command(["start", "wmplayer"], speech_queue, "Opening Windows Media Player.", "Failed to open music player")
        elif tag == "open_instagram":
            webbrowser.open("https://www.instagram.com")
            speech_queue.put("Opening Instagram.")
        elif tag == "open_facebook":
            webbrowser.open("https://www.facebook.com")
            speech_queue.put("Opening Facebook.")
        elif tag == "open_maps":
            webbrowser.open("https://www.google.com/maps") # Corrected URL
            speech_queue.put("Opening Google Maps.")
        elif tag == "open_email":
            webbrowser.open("https://www.gmail.com")
            speech_queue.put("Opening Gmail.")
        elif tag == "open_twitter":
            webbrowser.open("https://www.twitter.com")
            speech_queue.put("Opening X (formerly Twitter).")
        elif tag == "open_reddit":
            webbrowser.open("https://www.reddit.com")
            speech_queue.put("Opening Reddit.")
        elif tag == "open_amazon":
            webbrowser.open("https://www.amazon.in")
            speech_queue.put("Opening Amazon India.")
        elif tag == "open_linkedin":
            webbrowser.open("https://www.linkedin.com")
            speech_queue.put("Opening LinkedIn.")
        elif tag == "open_gallery":
            run_windows_command(["start", "ms-photos:"], speech_queue, "Opening photos.", "Failed to open gallery")
        elif tag == "open_file_explorer":
            run_windows_command("explorer", speech_queue, "Opening File Explorer.", "Failed to open File Explorer")
        elif tag == "open_cmd":
            run_windows_command("cmd", speech_queue, "Opening Command Prompt.", "Failed to open Command Prompt")
        elif tag == "open_powershell":
            run_windows_command("powershell", speech_queue, "Opening PowerShell.", "Failed to open PowerShell")
        elif tag == "open_word":
            run_windows_command(["start", "winword"], speech_queue, "Opening Microsoft Word.", "Failed to open Word")
        elif tag == "open_powerpoint":
            run_windows_command(["start", "powerpnt"], speech_queue, "Opening Microsoft PowerPoint.", "Failed to open PowerPoint")
        elif tag == "open_excel":
            run_windows_command(["start", "excel"], speech_queue, "Opening Microsoft Excel.", "Failed to open Excel")
        elif tag == "open_access":
            run_windows_command(["start", "access"], speech_queue, "Opening Microsoft Access.", "Failed to open Access")
        elif tag == "open_outlook":
            run_windows_command(["start", "outlook"], speech_queue, "Opening Microsoft Outlook.", "Failed to open Outlook")
        elif tag == "open_teams":
            run_windows_command(["start", "teams"], speech_queue, "Opening Microsoft Teams.", "Failed to open Teams")
        elif tag == "open_edge":
            run_windows_command(["start", "msedge"], speech_queue, "Opening Microsoft Edge.", "Failed to open Edge")
        elif tag == "open_chrome":
            run_windows_command(["start", "chrome"], speech_queue, "Opening Google Chrome.", "Failed to open Chrome")
        elif tag == "open_brave":
            run_windows_command(["start", "brave"], speech_queue, "Opening Brave Browser.", "Failed to open Brave")
        elif tag == "open_vscode":
            run_windows_command(["start", "code"], speech_queue, "Opening VS Code.", "Failed to open VS Code")
        elif tag == "open_task_manager":
            run_windows_command(["start", "taskmgr"], speech_queue, "Opening Task Manager.", "Failed to open Task Manager")
        elif tag == "open_calendar":
            webbrowser.open("https://calendar.google.com")
            speech_queue.put("Opening Google Calendar.")
        elif tag == "open_drive":
            webbrowser.open("https://drive.google.com")
            speech_queue.put("Opening Google Drive.")
        elif tag == "open_onenote":
            run_windows_command(["start", "onenote"], speech_queue, "Opening OneNote.", "Failed to open OneNote")
        elif tag == "check_battery":
            battery = psutil.sensors_battery()
            if battery:
                speech_queue.put(f"The battery percentage is {battery.percent}%")
                if battery.power_plugged:
                    speech_queue.put("And it is currently charging.")
                else:
                    speech_queue.put("And it is currently not charging.")
            else:
                speech_queue.put("Sorry, I couldn't get battery information.")
                logging.warning("Could not retrieve battery information.")
        elif tag == "check_date":
            date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            speech_queue.put(f"The date is {date}")
        elif tag == "check_weather":
            check_weather(speech_queue) # Passes speech_queue to weather function
        elif tag == "set_alarm":
            time_str = extract_alarm_time(query)
            if time_str:
                set_alarm(time_str, speech_queue)
            else:
                speech_queue.put("Sorry, I couldn't understand the time you mentioned for the alarm. Please try again with a specific time like '5:30 PM'.")
        elif tag == "search_google":
            search_query = extract_google_query(query)
            if search_query:
                url = f"https://www.google.com/search?q={webbrowser.quote(search_query)}" # Use quote for URL encoding
                webbrowser.open(url)
                speech_queue.put(f"Searching Google for {search_query}.")
            else:
                speech_queue.put("I couldn't extract a search query. What would you like to search for?")
        elif tag == "math_operation":
            speech_queue.put(math_operation(query))
        elif tag == "take_screenshot":
            take_screenshot(speech_queue)
        elif tag == "mute_volume":
            mute_volume()
            speech_queue.put("Volume muted.")
        elif tag == "unmute_volume":
            unmute_volume()
            speech_queue.put("Volume unmuted.")
        elif tag == "volume_up":
            pyautogui.press("volumeup")
            speech_queue.put("Volume up.")
        elif tag == "volume_down":
            pyautogui.press("volumedown")
            speech_queue.put("Volume down.")
        elif tag == "increase_brightness":
            current_brightness = sbc.get_brightness()[0]
            new_brightness = min(current_brightness + 10, 100)
            sbc.set_brightness(new_brightness)
            speech_queue.put(f"Brightness set to {new_brightness} percent.")
        elif tag == "decrease_brightness":
            current_brightness = sbc.get_brightness()[0]
            new_brightness = max(current_brightness - 10, 0)
            sbc.set_brightness(new_brightness)
            speech_queue.put(f"Brightness set to {new_brightness} percent.")
        elif tag == "lock_screen":
            if os.name == 'nt': # Check if OS is Windows
                ctypes.windll.user32.LockWorkStation()
                speech_queue.put("Locking screen.")
            else:
                speech_queue.put("Screen locking is only supported on Windows.")
                logging.warning("Attempted to lock screen on non-Windows OS.")
        elif tag == "shutdown":
            if os.name == 'nt':
                run_windows_command(["shutdown", "/s", "/t", "1"], speech_queue, "Shutting down the system.", "Failed to shut down")
            else:
                speech_queue.put("System shutdown is only supported on Windows.")
                logging.warning("Attempted to shut down on non-Windows OS.")
        elif tag == "restart":
            if os.name == 'nt':
                run_windows_command(["shutdown", "/r", "/t", "1"], speech_queue, "Restarting the system.", "Failed to restart")
            else:
                speech_queue.put("System restart is only supported on Windows.")
                logging.warning("Attempted to restart on non-Windows OS.")
        elif tag == "logout_user":
            if os.name == 'nt':
                run_windows_command(["shutdown", "-l"], speech_queue, "Logging out user.", "Failed to log out")
            else:
                speech_queue.put("User logout is only supported on Windows.")
                logging.warning("Attempted to log out on non-Windows OS.")
        elif tag == "diagnose_symptoms":
            symptoms_query = query.replace("diagnose symptoms", "").strip()
            diagnose_symptoms(symptoms_query, speech_queue) # Pass speech_queue
        else:
            speech_queue.put("I'm sorry, I don't understand that command or the feature is not yet implemented.")
            logging.info(f"Unrecognized tag: {tag}")
    except Exception as e:
        speech_queue.put(f"An unexpected error occurred during the task: {str(e)}")
        logging.critical(f"Critical error in perform_task for tag '{tag}': {e}", exc_info=True)