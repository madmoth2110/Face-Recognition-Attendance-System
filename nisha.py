import cv2

# -------------------------------
# 1. Names and IDs
# -------------------------------

names = {
    0: "Nisha"
}

# -------------------------------
# 2. Load trained model
# -------------------------------

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# -------------------------------
# 3. Load face detector
# -------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -------------------------------
# 4. Start webcam
# -------------------------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera could not be opened")
    exit()

print("Camera started...")
print("Press ESC to exit.")

# -------------------------------
# 5. Recognition loop
# -------------------------------

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read camera")
        break

    # Convert camera image to grayscale
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

    # -------------------------------
    # 6. Recognize each face
    # -------------------------------

    for (x, y, w, h) in faces:

        face_roi = gray[y:y+h, x:x+w]

        id_, distance = recognizer.predict(face_roi)

        # LBPH: lower distance = better match
        if distance < 75:

            name = names.get(
                id_,
                "Unknown"
            )

            color = (0, 255, 0)
            status = "Recognized"

        else:

            name = "Unknown"

            color = (0, 0, 255)
            status = "Unknown"

        # -------------------------------
        # 7. Draw face rectangle
        # -------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # -------------------------------
        # 8. Display name
        # -------------------------------

        cv2.putText(
            frame,
            name,
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        # Display status
        cv2.putText(
            frame,
            status,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Display LBPH distance
        cv2.putText(
            frame,
            f"Distance: {distance:.1f}",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Print result in terminal
        print(
            f"Name: {name} | ID: {id_} | "
            f"Distance: {distance:.2f}"
        )

    # -------------------------------
    # 9. Show camera
    # -------------------------------

    cv2.imshow(
        "Face Recognition System",
        frame
    )

    # ESC key
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

# -------------------------------
# 10. Close everything
# -------------------------------

camera.release()
cv2.destroyAllWindows()