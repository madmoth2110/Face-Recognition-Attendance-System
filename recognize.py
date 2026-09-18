import cv2
from deepface import DeepFace
import csv
from datetime import datetime
import os

attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])

marked_today = set()


def mark_attendance(name):
    today = datetime.now().strftime("%d-%m-%Y")

    if name == "Unknown" or name in marked_today:
        return

    current_time = datetime.now().strftime("%I:%M:%S %p")

    with open(attendance_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, today, current_time])

    marked_today.add(name)

    print(f"Attendance marked: {name} | {today} | {current_time}")


names = {
    0: "Hema",
    1: "Nisha"
}


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera error")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            emotion = result[0]["dominant_emotion"]

        except:
            emotion = "Unknown"

        face_roi = gray[y:y+h, x:x+w]

        id_, confidence = recognizer.predict(face_roi)

        if confidence < 100:
            name = names.get(id_, "Unknown")
            color = (0, 255, 0)

            mark_attendance(name)

        else:
            name = "Unknown"
            color = (0, 0, 255)

        print(
            f"Name: {name} | Emotion: {emotion} | "
            f"Confidence: {confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.0f}",
            (x, y + h + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow(
        "Face Recognition System",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()