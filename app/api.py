import base64
import json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.session import LivenessSession
from app.detector import ensure_model, build_landmarker
import time

app = FastAPI()


def decode_frame(data: str):
    """
    Client sends each frame as a base64-encoded JPEG.
    We decode it back into a numpy array OpenCV can work with.
    """
    raw = base64.b64decode(data)
    buf = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return frame


def run_mediapipe(landmarker, frame, timestamp_ms):
    """
    Run MediaPipe on a single frame and return landmarks or None.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    if results.face_landmarks:
        return results.face_landmarks[0]
    return None


@app.websocket("/ws/liveness")
async def liveness_ws(websocket: WebSocket):
    """
    One WebSocket connection = one liveness session.

    The client sends frames as JSON:
        { "frame": "<base64 encoded JPEG>" }

    The server responds after every frame with the current state:
        {
            "state": "BLINK",
            "prompt": "Please blink",
            "order": ["BLINK", "TURN_LEFT", "TURN_RIGHT"],
            "index": 0,
            "time_left": 5.2,
            "checks": { ... },
            "result": null
        }

    When state becomes "DONE", result is populated and the
    server closes the connection.
    """
    await websocket.accept()
    ensure_model()

    session = LivenessSession()

    with build_landmarker() as landmarker:
        try:
            while True:
                raw = await websocket.receive_text()
                message = json.loads(raw)
                frame = decode_frame(message["frame"])

                timestamp_ms = int(time.time() * 1000)
                landmarks = run_mediapipe(landmarker, frame, timestamp_ms)
                status = session.process(landmarks)

                await websocket.send_text(json.dumps(status))

                if status["state"] == "DONE" and status["result"] is not None:
                    break

        except WebSocketDisconnect:
            pass

    await websocket.close()