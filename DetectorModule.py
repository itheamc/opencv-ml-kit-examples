import cv2
import mediapipe as mp

# Creating drawing_utils reference from the MediaPipe.solutions
mp_drawings = mp.solutions.drawing_utils
mp_drawing_specs = mp_drawings.DrawingSpec(thickness=1, circle_radius=1)


# Function to convert the BGR image to RGB image
def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------
# ----------------------HAND DETECTOR CLASS ---------------------------------
# ----------------------------------------------------------------------------

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mp_hands = mp.solutions.hands

        # MediaPipe Drawing utils to draw the landmarks and other shape on the image
        self.hands = self.mp_hands.Hands(static_image_mode, max_num_hands, min_detection_confidence,
                                         min_tracking_confidence)

    # Function to detect the hands and drawing accordingly
    def detect_hands(self, image):
        results = self.hands.process(bgr_to_rgb(image))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawings.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return image


# ----------------------------------------------------------------------------
# ----------------------POSE DETECTOR CLASS ---------------------------------
# ----------------------------------------------------------------------------

class PoseDetector:
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose

        # MediaPipe Drawing utils to draw the landmarks and other shape on the image
        self.pose = self.mp_pose.Pose(static_image_mode,
                                      model_complexity,
                                      smooth_landmarks,
                                      min_detection_confidence,
                                      min_tracking_confidence)

    # Function to detect pose
    def detect_pose(self, image):
        results = self.pose.process(bgr_to_rgb(image))

        if results.pose_landmarks:
            mp_drawings.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return image


# ---------------------------------------------------------------------------
# ---------------------------FACE DETECTOR CLASS---------------------------
# ---------------------------------------------------------------------------
class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence)

    # Function to detect faces
    def detect_faces(self, image):
        results = self.face_detection.process(bgr_to_rgb(image))

        if results.detections:
            for detection in results.detections:
                mp_drawings.draw_detection(image, detection)

        return image


# ---------------------------------------------------------------------------
# ---------------------------FACE MESH DRAWER---------------------------
# ---------------------------------------------------------------------------
class FaceMeshDrawer:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5
                 ):

        self.mp_face_mesh = mp.solutions.face_mesh

        # Creating the instance of the FaceMesh()
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode,
            max_num_faces,
            min_detection_confidence,
            min_tracking_confidence)

    # Function to draw facemask
    def draw_face_mesh(self, image):
        # Processing image to get the landmarks
        results = self.face_mesh.process(bgr_to_rgb(image))

        # If results is not empty and have face_landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawings.draw_landmarks(image=image,
                                           landmark_list=face_landmarks,
                                           connections=self.mp_face_mesh.FACE_CONNECTIONS,
                                           landmark_drawing_spec=mp_drawing_specs,
                                           connection_drawing_spec=mp_drawing_specs)

        return image
