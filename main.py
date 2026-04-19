import cv2
from config import CAMERA_INDEX, FRAME_RESIZE, TOLERANCE
from camera.camera_stream import CameraStream
from recognition.encoder import encode_faces
from recognition.recognizer import FaceRecognizer
from database.face_database import load_database
from utils.drawing import draw_faces

def main():
    print("[INFO] Loading face database...")
    known_encodings, known_names = load_database()

    recognizer = FaceRecognizer(known_encodings, known_names, TOLERANCE)
    camera = CameraStream(CAMERA_INDEX)

    print("[INFO] Starting camera...")

    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)

        locations, encodings = encode_faces(small_frame)
        names = recognizer.recognize(encodings)

        frame = draw_faces(frame, locations, names)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()

if __name__ == "__main__":
    main()