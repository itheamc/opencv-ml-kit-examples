import cv2
from DetectorModule import FaceDetector

capture = cv2.VideoCapture('http://192.168.0.100:4747/video')
capture.set(3, 1920)
capture.set(4, 1080)

# mp_face = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
#
# with mp_face.FaceDetection() as face_detection:
#     while cam.isOpened():
#         success, img = cam.read()
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = face_detection.process(rgb_img)
#
#         if results.detections:
#             for detection in results.detections:
#                 mp_drawing.draw_detection(img, detection)
#
#         cv2.imshow('Face Detector -- By Amit', img)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break


face_detector = FaceDetector()

while capture.isOpened():
    success, img = capture.read()

    if not success:
        continue

    img = face_detector.detect_faces(img)

    cv2.imshow('Face Detector - By itheamc', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

