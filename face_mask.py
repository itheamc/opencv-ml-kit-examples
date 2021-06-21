import cv2
import mediapipe as mp
from DetectorModule import FaceMeshDrawer, HandDetector, FaceDetector, PoseDetector

# Capturing Video Using FaceCam
capture = cv2.VideoCapture(0)
capture.set(3, 1280)    # Width
capture.set(4, 720)     # Height

# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Creating FaceMesh object as default options
# static_image_mode=False,
# max_num_faces=1,
# min_detection_confidence=0.5,
# min_tracking_confidence=0.5
# with mp_face_mesh.FaceMesh(max_num_faces=2) as face_mesh:
#     while cam.isOpened():
#         success, img = cam.read()
#
#         if not success:
#             continue
#
#         # Converting BGR image to RGB
#         rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         # Processing image to get the landmarks
#         results = face_mesh.process(rgb_image)
#
#         # If results is not empty and have face_landmarks
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 mp_drawing.draw_landmarks(image=img,
#                                           landmark_list=face_landmarks,
#                                           connections=mp_face_mesh.FACE_CONNECTIONS,
#                                           landmark_drawing_spec=mp_drawing_specs,
#                                           connection_drawing_spec=mp_drawing_specs)
#
#         cv2.imshow('FaceMesh', img)
#         cv2.waitKey(1)

face_mesh_drawer = FaceMeshDrawer()

while capture.isOpened():
    success, img = capture.read()

    if not success:
        continue

    img = face_mesh_drawer.draw_face_mesh(img)

    cv2.imshow('FaceMesh', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

