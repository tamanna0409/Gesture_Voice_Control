import cv2
import numpy as np
from gesture_control import process_gestures
from voice_control import listen_for_command

cap = cv2.VideoCapture(0)

# Function to check light conditions
def check_light_condition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness > 50  # True if bright enough

# Check if camera is available
if cap.isOpened():
    print(" Camera detected. Checking light conditions...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if check_light_condition(frame):  # Use gesture control if light is available
            frame = process_gestures(frame)
            cv2.putText(frame, "Using Gesture Control", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:  # Use voice commands if it's dark
            cv2.putText(frame, "Using Voice Control", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            listen_for_command()

        cv2.imshow("Gesture & Voice Control", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):  # Press 'Q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print(" No camera found. Switching to voice control mode.")
    while True:
        listen_for_command()
