import cv2
import os
import numpy as np

path = "employees"

faces = []
ids = []
names = {}

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

current_id = 0

for file in os.listdir(path):

    if file.endswith(".jpg"):

        name = file.split(".")[0]

        names[current_id] = name

        image = cv2.imread(
            os.path.join(path, file),
            cv2.IMREAD_GRAYSCALE
        )

        detected_faces = face_detector.detectMultiScale(
            image
        )

        for (x, y, w, h) in detected_faces:

            faces.append(
                image[y:y+h, x:x+w]
            )

            ids.append(current_id)

        current_id += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(ids)
)

recognizer.save("trainer.yml")

print("Training completed")
print(names)