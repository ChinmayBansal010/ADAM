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
import ctypes

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

def mute_volume():
    volume.SetMute(1, None)  # 1 = mute

def unmute_volume():
    volume.SetMute(0, None)  # 0 = unmute

def is_muted():
    return volume.GetMute() == 1

def take_screenshot(speech_queue):
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
        
def check_weather(speech_queue=None):
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

def perform_task(tag, query=None,speech_queue=None):
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
