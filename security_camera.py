import cv2
import time

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

ret, frame1 = camera.read()
ret, frame2 = camera.read()

while True:

    diff = cv2.absdiff(frame1, frame2)

    gray_motion = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray_motion, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur,
        20,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_found = False

    for contour in contours:

        if cv2.contourArea(contour) < 1000:
            continue

        motion_found = True

    gray = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        1.1,
        5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame1,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame1,
            "FACE",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    if motion_found:

        cv2.putText(
            frame1,
            "MOTION ALERT",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        filename = f"alert_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame1)

    cv2.putText(
        frame1,
        f"Faces: {len(faces)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.imshow(
        "AI Security Camera",
        frame1
    )

    frame1 = frame2
    ret, frame2 = camera.read()

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()