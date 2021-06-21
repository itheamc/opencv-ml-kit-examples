import cv2
from DetectorModule import PoseDetector

capture = cv2.VideoCapture(0)

pose_detector = PoseDetector()

while capture.isOpened():
    success, img = capture.read()

    if not success:
        continue

    img = pose_detector.detect_pose(img)

    cv2.imshow('Pose Detector -- By Amit', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

