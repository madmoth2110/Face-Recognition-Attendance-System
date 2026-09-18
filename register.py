import cv2
import os

name = input("Enter employee name: ").strip()

if not os.path.exists("employees"):
    os.makedirs("employees")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera started.")
print("Look at the camera.")
print("Press S to capture a sample.")
print("Press ESC to cancel.")

count = 0

while True:

    success, frame = camera.read()

    if not success:
        print("Camera error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {count}/10",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Employee: {name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Employee Registration", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        if len(faces) == 0:
            print("No face detected. Try again.")
            continue

        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]

        filename = f"employees/{name}_{count}.jpg"

        cv2.imwrite(filename, face)

        count += 1

        print(f"Saved: {filename}")

        if count >= 10:
            print("10 face samples saved successfully.")
            break

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()