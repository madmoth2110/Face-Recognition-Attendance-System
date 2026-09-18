from deepface import DeepFace
import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera error")
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion = result[0]["dominant_emotion"]

        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    except Exception as e:
        print("Emotion detection error:", e)

    cv2.imshow("Emotion Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()