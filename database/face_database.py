import os
import face_recognition

def load_database(dataset_path="dataset"):
    known_encodings = []
    known_names = []

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)

    return known_encodings, known_names