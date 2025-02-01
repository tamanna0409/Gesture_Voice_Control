import cv2

def check_light_condition():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(" Could not capture image.")
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = gray.mean()
    print(f" Average Brightness: {avg_brightness}")

    return avg_brightness > 50  # Threshold to determine if light is sufficient
