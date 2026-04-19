import face_recognition
import numpy as np

def encode_faces(image):
    rgb = np.ascontiguousarray(image[:, :, ::-1])

    locations = face_recognition.face_locations(rgb)

    encodings = []
    if locations:
        encodings = face_recognition.face_encodings(rgb, locations)

    return locations, encodings