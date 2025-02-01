'''import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Audio Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

def process_hand_gestures():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate Euclidean distance
                distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

                # Map distance to brightness and volume
                brightness = int(distance * 100)
                volume_level = min(max(distance * 10, 0), 1)

                sbc.set_brightness(brightness)
                volume.SetMasterVolumeLevelScalar(volume_level, None)

                print(f" Brightness: {brightness}, 🔊 Volume: {volume_level}")

        cv2.imshow("Hand Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()'''




'''import cv2
import mediapipe as mp
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get system volume control
def get_volume_control():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))

volume = get_volume_control()

# Function to adjust brightness
def set_brightness(level):
    sbc.set_brightness(level)
    print(f"Brightness set to {level}%")

# Function to adjust volume
def set_volume(level):
    volume.SetMasterVolumeLevelScalar(level / 100, None)
    print(f"Volume set to {level}%")

# Function to process gestures
def process_gestures(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]  # Index Finger Tip
            thumb_tip = hand_landmarks.landmark[4]  # Thumb Tip

            height, width, _ = frame.shape
            x1, y1 = int(index_tip.x * width), int(index_tip.y * height)
            x2, y2 = int(thumb_tip.x * width), int(thumb_tip.y * height)

            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # Adjust brightness based on distance
            if 50 < distance < 100:
                set_brightness(30)
            elif 100 < distance < 150:
                set_brightness(60)
            elif distance > 150:
                set_brightness(100)

            # Adjust volume based on hand position
            if y1 < height // 3:
                set_volume(min(100, volume.GetMasterVolumeLevelScalar() * 100 + 10))
            elif y1 > (2 * height) // 3:
                set_volume(max(0, volume.GetMasterVolumeLevelScalar() * 100 - 10))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame'''


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

# Get volume range (-65.25 dB to 0 dB)
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


'''import cv2
import mediapipe as mp
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# Function to control brightness (mocked)
def set_brightness(level):
    # Logic to control system brightness (this will vary depending on the OS)
    print(f"Brightness set to: {level}%")

# Function to control volume (mocked)
def set_volume(level):
    # Logic to control system volume (this will vary depending on the OS)
    print(f"Volume set to: {level}%")

# Function to calculate distance between thumb and index for brightness
def calculate_brightness(thumb_index_distance):
    # This is just an example, you can scale or adjust as per your preference
    return min(100, max(0, int(thumb_index_distance * 10)))  # Scale accordingly

# Function to calculate proximity for volume (using hand's distance to camera)
def calculate_volume(hand_distance):
    # Adjust the scale as per how far/close the hand is to the screen
    return min(100, max(0, int(hand_distance * 100)))  # Scale accordingly

# Gesture recognition loop using OpenCV
def gesture_control():
    cap = cv2.VideoCapture(0)  # Initialize the webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hands = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')  # Ensure you have hand classifier

        # Detect hands in the frame
        hands_detected = hands.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in hands_detected:
            hand_area = frame[y:y+h, x:x+w]
            # Detect key points for thumb and index, calculate distance between them for brightness
            thumb_index_distance = calculate_thumb_index_distance(hand_area)
            
            # Detect hand proximity to adjust volume
            hand_distance = calculate_hand_proximity(x, y, w, h)

            # Calculate brightness and volume
            brightness_level = calculate_brightness(thumb_index_distance)
            volume_level = calculate_volume(hand_distance)

            # Apply brightness and volume settings
            set_brightness(brightness_level)
            set_volume(volume_level)

        # Display the frame with the hand detection (optional)
        cv2.imshow('Gesture Control', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Helper function to calculate the distance between thumb and index finger
def calculate_thumb_index_distance(hand_area):
    # Placeholder for actual thumb and index detection logic
    # You would use something like the hand landmarks from mediapipe or a similar approach
    # Return a mock distance value for now (e.g., distance from 0 to 10)
    return 5  # Example value

# Helper function to calculate hand proximity (distance from camera)
def calculate_hand_proximity(x, y, w, h):
    # A basic mock function that uses the size of the hand bounding box as a proxy for proximity
    # You can refine this to be more accurate depending on the camera setup and distance
    return 1 / (x + y + w + h)  # Simple proxy for proximity; adjust as needed

if __name__ == "__main__":
    gesture_control()'''

