import cv2

def check_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Camera not available. Using voice commands only.")
        return False
    cap.release()
    print(" Camera available.")
    return True
