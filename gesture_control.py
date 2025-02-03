import cv2
import mediapipe as mp
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize volume control
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
    level = max(0, min(level, 100))  # Keep it in range (0 - 100)
    volume_db = vol_min + (level / 100) * (vol_max - vol_min)
    volume.SetMasterVolumeLevel(volume_db, None)
    print(f"Volume set to {level}%")

# Function to detect gestures and adjust brightness/volume
def process_gestures(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            height, width, _ = frame.shape
            fingers = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]  # Thumb, Index, Middle, Ring, Pinky

            # Calculate distances
            spread_distance = abs(fingers[0].x - fingers[4].x) * width  # Thumb to pinky distance
            two_finger_distance = abs(fingers[1].y - fingers[2].y) * height  # Index to middle distance
            fist_closeness = abs(fingers[0].y - fingers[1].y) * height  # Thumb to index distance

            # Gesture Detection
            if spread_distance > 200:  # Spread fingers wide 🖐
                set_brightness(100)
            elif fist_closeness < 40:  # Close fist ✊
                set_brightness(10)
            elif two_finger_distance > 50:  # Two fingers up ✌️
                set_volume(100)
            elif spread_distance < 50:  # Thumb + Pinky (Shaka Sign) 🤙
                set_volume(10)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame
