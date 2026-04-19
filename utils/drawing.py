import cv2

def draw_faces(frame, locations, names):
    for (top, right, bottom, left), name in zip(locations, names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

    return frame