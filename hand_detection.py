import cv2
from DetectorModule import HandDetector

capture = cv2.VideoCapture("http://192.168.0.100:4747/video")
capture.set(3, 1280)
capture.set(4, 720)

# Creating the instance of the HandDetector class
hand_detector = HandDetector()

while capture.isOpened():
    success, img = capture.read()

    if not success:
        continue

    # Detecting hands and drawing the landmarks on image
    img = hand_detector.detect_hands(image=img)

    cv2.imshow('Hand Detector -- By Amit', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
