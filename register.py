import cv2
import os

# Ask for employee name
name = input("Enter employee name: ")

# Create employees folder if it doesn't exist
if not os.path.exists("employees"):
    os.makedirs("employees")

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Start camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera started.")
print("Look at the camera.")
print("Press S to register.")
print("Press ESC to cancel.")

while True:

    success, frame = camera.read()

    if not success:
        print("Camera error")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # Draw face boxes
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
            "Face detected - Press S",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Show employee name
    cv2.putText(
        frame,
        f"Employee: {name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Employee Registration",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Press S to save
    if key == ord('s'):

        if len(faces) == 0:
            print("No face detected. Try again.")
            continue

        # Take the first detected face
        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]

        filename = f"employees/{name}.jpg"

        cv2.imwrite(
            filename,
            face
        )

        print(f"Saved: {filename}")
        break

    # Press ESC
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
