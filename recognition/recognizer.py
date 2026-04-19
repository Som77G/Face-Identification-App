import face_recognition
import numpy as np

class FaceRecognizer:
    def __init__(self, known_encodings, known_names, tolerance=0.5):
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.tolerance = tolerance

    def recognize(self, encodings):
        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(
                self.known_encodings, encoding, self.tolerance
            )
            face_distances = face_recognition.face_distance(
                self.known_encodings, encoding
            )

            name = "Unknown"

            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                    name = self.known_names[best_match]

            names.append(name)

        return names