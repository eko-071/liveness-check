import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.session import LivenessSession, PROMPTS, ALL_CHALLENGES
import urllib.request
import os
import time

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

LABEL = {
    "BLINK": "Blink",
    "TURN_LEFT": "Turn left",
    "TURN_RIGHT": "Turn right",
}


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def build_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.FaceLandmarker.create_from_options(options)


def draw_overlay(frame, status):
    h, w, _ = frame.shape
    state = status["state"]
    checks = status["checks"]
    result = status["result"]
    order = status["order"]
    index = status["index"]
    time_left = status["time_left"]
    prompt = status["prompt"]

    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.putText(frame, f"State: {state}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if state not in ("CALIBRATE", "DONE"):
        timer_color = (0, 255, 0) if time_left > 3 else (0, 100, 255)
        cv2.putText(frame, f"{time_left:.1f}s", (w - 80, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2)

        for i, ch in enumerate(order):
            color = (
                (0, 220, 0)   if i < index else
                (0, 220, 220) if i == index else
                (80, 80, 80)
            )
            cv2.circle(frame, (w - 200 + i * 30, 20), 7, color, -1)

    if prompt:
        cv2.putText(frame, prompt, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    y = 130
    for ch in order:
        info = checks.get(ch, {})
        passed = info.get("passed", False)
        timed_out = info.get("timed_out", False)
        symbol, color = (
            ("PASS", (0, 220, 0))    if passed else
            ("FAIL", (0, 60, 220))   if timed_out else
            ("...",  (100, 100, 100))
        )
        cv2.putText(frame, f"{symbol} {LABEL[ch]}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 28

    if result:
        status_text = result["status"].upper()
        color = (0, 220, 0) if status_text == "LIVE" else (0, 0, 220)
        cv2.putText(frame, f"{status_text}  score: {result['score']}", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        bd = result["breakdown"]
        summary = f"base:{bd['weighted_score']}  consistency:+{bd['consistency_bonus']}  timeout:-{bd['timeout_penalty']}"
        cv2.putText(frame, summary, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def run_liveness_check():
    ensure_model()
    session = LivenessSession()
    cap = cv2.VideoCapture(0)

    with build_landmarker() as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect_for_video(mp_image, int(now * 1000))

            landmarks = (results.face_landmarks[0] if results.face_landmarks else None)

            status = session.process(landmarks)

            if status["state"] == "DONE" and status["result"]:
                r = status["result"]
                print(f"Result : {r['status'].upper()}")
                print(f"Score : {r['score']}")
                print(f"Breakdown : {r['breakdown']}")
                print(f"Checks : {r['checks']}")

            for lm in (results.face_landmarks[0] if results.face_landmarks else []):
                h, w, _ = frame.shape
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

            draw_overlay(frame, status)
            cv2.imshow("Liveness Check", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                session = LivenessSession()
                print("Restarted.")

    cap.release()
    cv2.destroyAllWindows()