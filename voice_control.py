'''import speech_recognition as sr
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Audio Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

def recognize_voice_command():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print(" Listening for voice command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            print(f" Recognized: {command}")

            if "increase brightness" in command:
                sbc.set_brightness(min(sbc.get_brightness()[0] + 10, 100))
            elif "decrease brightness" in command:
                sbc.set_brightness(max(sbc.get_brightness()[0] - 10, 0))
            elif "increase volume" in command:
                volume.SetMasterVolumeLevelScalar(min(volume.GetMasterVolumeLevelScalar() + 0.1, 1), None)
            elif "decrease volume" in command:
                volume.SetMasterVolumeLevelScalar(max(volume.GetMasterVolumeLevelScalar() - 0.1, 0), None)
        
        except sr.UnknownValueError:
            print(" Could not understand the command.")
        except sr.RequestError:
            print("Error connecting to recognition service.")

if __name__ == "__main__":
    while True:
        recognize_voice_command()'''

import speech_recognition as sr
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import re

# Get system volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
vol_min, vol_max = volume.GetVolumeRange()[:2]

# Function to adjust brightness
def set_brightness(level):
    sbc.set_brightness(level)
    print(f"Brightness set to {level}%")

# Function to adjust volume
def set_volume(level):
    level = max(0, min(level, 100))
    volume_db = vol_min + (level / 100) * (vol_max - vol_min)
    volume.SetMasterVolumeLevel(volume_db, None)
    print(f"Volume set to {level}%")

# Function to process voice command
def process_voice_command(command):
    match = re.search(r"(\d+)", command)
    if match:
        value = int(match.group(1))

        if "increase brightness" in command:
            set_brightness(min(100, sbc.get_brightness()[0] + value))
        elif "decrease brightness" in command:
            set_brightness(max(0, sbc.get_brightness()[0] - value))
        elif "set brightness" in command:
            set_brightness(max(0, min(value, 100)))
        elif "increase volume" in command:
            set_volume(min(100, volume.GetMasterVolumeLevelScalar() * 100 + value))
        elif "decrease volume" in command:
            set_volume(max(0, volume.GetMasterVolumeLevelScalar() * 100 - value))
        elif "set volume" in command:
            set_volume(max(0, min(value, 100)))

# Function to listen for commands
def listen_for_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            process_voice_command(command)
        except sr.UnknownValueError:
            print("Sorry, I didn't understand.")
        except sr.RequestError:
            print("Could not request results, check your internet.")
