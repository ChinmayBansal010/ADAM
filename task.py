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

load_dotenv()
GEMINI_ACCESS_KEY = os.getenv("GEMINI_ACCESS_KEY")


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

def mute_volume():
    volume.SetMute(1, None)

def unmute_volume():
    volume.SetMute(0, None)

def is_muted():
    return volume.GetMute() == 1

def take_screenshot(speech_queue):
    try:
        pyautogui.screenshot().save("screenshot.png")
        speech_queue.put("Screenshot taken.")
    except Exception as e:
        speech_queue.put(f"Sorry, there was an error taking the screenshot: {str(e)}")

def _safe_eval(expr):
    # A very simple, safe evaluator for two numbers and one operator
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        'x': operator.mul,
    }
    # Find numbers (including decimals) and operators
    numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', expr)]
    op_chars = re.findall(r'[\+\-\*\/x]', expr)
    
    if len(numbers) == 2 and len(op_chars) == 1:
        op_func = ops.get(op_chars[0])
        if op_func:
            if op_chars[0] == '/' and numbers[1] == 0:
                return "Cannot divide by zero."
            return op_func(numbers[0], numbers[1])
    return None

def math_operation(query):
    query = query.lower().replace('what is', '').replace('calculate', '').strip()
    query = query.replace('plus', '+').replace('add', '+')
    query = query.replace('minus', '-').replace('subtract', '-')
    query = query.replace('times', '*').replace('multiplied by', '*')
    query = query.replace('divided by', '/').replace('divide', '/')
    
    result = _safe_eval(query)
    if result is not None:
        return f"The result is {round(result, 4) if isinstance(result, float) else result}"
    return "Sorry, I couldn't perform that calculation."

def set_alarm(time_str, speech_queue, alarm_sound_path="alarm_sound.mp3"):
    try:
        alarm_time_obj = datetime.datetime.strptime(time_str, "%I:%M %p").time()
        now = datetime.datetime.now()
        alarm_time = datetime.datetime.combine(now.date(), alarm_time_obj)
        if alarm_time <= now:
            alarm_time += datetime.timedelta(days=1)
        time_until_alarm = (alarm_time - now).total_seconds()

        def trigger_alarm():
            try:
                play(AudioSegment.from_file(alarm_sound_path))
            except Exception as e:
                speech_queue.put(f"Failed to play alarm: {str(e)}")

        Timer(time_until_alarm, trigger_alarm).start()
        speech_queue.put(f"Alarm set for {time_str}.")
    except ValueError:
        speech_queue.put("Sorry, there was an error setting the alarm. Please provide a valid time format.")

def check_weather(speech_queue=None):
    try:
        url = "https://www.google.com/search?q=weather"
        headers = {"User-Agent": "Mozilla/5.0"}
        soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
        temp = soup.find('div', class_="BNeawe iBp4i AP7Wnd").text
        condition = soup.find('div', class_="BNeawe tAd8D AP7Wnd").text
        speech_queue.put(f"The current temperature is {temp} and the weather condition is {condition}.")
    except:
        speech_queue.put("Sorry, I couldn't fetch the weather information at the moment.")

def extract_alarm_time(message):
    match = re.search(r"(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.|am|pm)?", message, re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        ampm = match.group(3)
        if ampm:
            if "p" in ampm.lower() and hour != 12:
                hour += 12
            elif "a" in ampm.lower() and hour == 12:
                hour = 0
        return f"{hour}:{minute:02d} {'AM' if 'a' in ampm.lower() else 'PM'}"
    return None

def extract_google_query(message):
    keywords = [
        r"search google for", r"google search for", r"search for", r"search google", r"google search", r"google for",
        r"can you google", r"could you search", r"please google", r"find information on", r"look up", r"search", r"google",
        r"find", r"can you look up", r"could you look up", r"i need info on", r"i want to know about", r"tell me about",
        r"can you find me", r"what is", r"who is", r"what are", r"how to", r"how do i"
    ]
    msg = re.sub(r'[^\w\s]', '', message.lower())
    for kw in keywords:
        match = re.search(rf"{kw}\s+(.*)", msg)
        if match:
            return match.group(1).strip()
    return msg

def diagnose_symptoms(symptoms):
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

        response = model.generate_content(prompt)
        clean_text = clean_response(response.text)
        return clean_text

    except Exception as e:
        return f"Failed to diagnose symptoms: {str(e)}"
    

def clean_response(text):
    text = re.sub(r"[#*`~•→⇒➤▶️▪️✅➔➥➽➧➠➤➢➣➞➟➡️➩➫➬➭➯➲➳➵➸➺➻➼➽➾]", "", text)
    text = re.sub(r"^\s*[-–—]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Final trim
    return text.strip()

def perform_task(tag, query=None, speech_queue=None):
    try:
        print(f"Tag received: {tag}")

        if tag == "open_calculator":
            subprocess.Popen("calc")
        elif tag == "open_browser":
            webbrowser.open("https://www.google.com")
        elif tag == "open_notepad":
            subprocess.Popen("notepad")
        elif tag == "tell_time":
            speech_queue.put(datetime.datetime.now().strftime("The time is %I:%M %p"))
        elif tag == "open_youtube":
            webbrowser.open("https://www.youtube.com")
        elif tag == "open_camera":
            subprocess.Popen(["start", "microsoft.windows.camera:"], shell=True)
        elif tag == "open_settings":
            subprocess.Popen(["start", "ms-settings:"], shell=True)
        elif tag == "open_whatsapp":
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
            subprocess.Popen(["start", "ms-photos:"], shell=True)
        elif tag == "open_file_explorer":
            subprocess.Popen("explorer")
        elif tag == "open_cmd":
            subprocess.Popen("cmd")
        elif tag == "open_powershell":
            subprocess.Popen("powershell")
        elif tag == "open_word":
            subprocess.Popen(["start", "winword"], shell=True)
        elif tag == "open_powerpoint":
            subprocess.Popen(["start", "powerpnt"], shell=True)
        elif tag == "open_excel":
            subprocess.Popen(["start", "excel"], shell=True)
        elif tag == "open_access":
            subprocess.Popen(["start", "access"], shell=True)
        elif tag == "open_outlook":
            subprocess.Popen(["start", "outlook"], shell=True)
        elif tag == "open_teams":
            subprocess.Popen(["start", "teams"], shell=True)
        elif tag == "open_edge":
            subprocess.Popen(["start", "msedge"], shell=True)
        elif tag == "open_chrome":
            subprocess.Popen(["start", "chrome"], shell=True)
        elif tag == "open_brave":
            subprocess.Popen(["start", "brave"], shell=True)
        elif tag == "open_vscode":
            subprocess.Popen(["start", "code"], shell=True)
        elif tag == "open_task_manager":
            subprocess.Popen(["start", "taskmgr"], shell=True)
        elif tag == "open_calendar":
            webbrowser.open("https://calendar.google.com")
        elif tag == "open_drive":
            webbrowser.open("https://drive.google.com")
        elif tag == "open_onenote":
            subprocess.Popen(["start", "onenote"], shell=True)
        elif tag == "check_battery":
            battery = psutil.sensors_battery()
            speech_queue.put(f"The battery percentage is {battery.percent}%")
        elif tag == "check_date":
            date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            speech_queue.put(f"The date is {date}")
        elif tag == "check_weather":
            check_weather(speech_queue)
        elif tag == "set_alarm":
            time_str = extract_alarm_time(query)
            if time_str:
                set_alarm(time_str, speech_queue)
            else:
                speech_queue.put("Sorry, I couldn't understand the time you mentioned.")
        elif tag == "search_google":
            url = f"https://www.google.com/search?q={extract_google_query(query)}"
            webbrowser.open(url)
        elif tag == "math_operation":
            speech_queue.put(math_operation(query))
        elif tag == "take_screenshot":
            take_screenshot(speech_queue)
        elif tag == "mute_volume":
            if is_muted():
                speech_queue.put("Volume is already muted.")
            else:
                mute_volume()
        elif tag == "unmute_volume":
            if not is_muted():
                speech_queue.put("Volume is already unmuted.")
            else:
                unmute_volume()
        elif tag == "volume_up":
            pyautogui.press("volumeup")
        elif tag == "volume_down":
            pyautogui.press("volumedown")
        elif tag == "increase_brightness":
            sbc.set_brightness(min(sbc.get_brightness()[0] + 10, 100))
        elif tag == "decrease_brightness":
            sbc.set_brightness(max(sbc.get_brightness()[0] - 10, 0))
        elif tag == "lock_screen":
            ctypes.windll.user32.LockWorkStation()
        elif tag == "shutdown":
            subprocess.Popen(["shutdown", "/s", "/t", "1"])
        elif tag == "restart":
            subprocess.Popen(["shutdown", "/r", "/t", "1"])
        elif tag == "logout_user":
            subprocess.Popen(["shutdown", "-l"])
        elif tag == "diagnose_symptoms":
            result = diagnose_symptoms(query)
            speech_queue.put(result)
        else:
            speech_queue.put("I'm sorry, I don't understand that command.")
    except Exception as e:
        speech_queue.put(f"Task failed: {str(e)}")
