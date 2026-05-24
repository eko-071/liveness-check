import base64
import json
import time
import uuid
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from app.session import LivenessSession
from app.detector import ensure_model, build_landmarker

app = FastAPI()

session_store = {}
SESSION_TIMEOUT = 60

def decode_frame(data: str):
    raw = base64.b64decode(data)
    buf = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def run_mediapipe(landmarker, frame, timestamp_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    if results.face_landmarks:
        return results.face_landmarks[0]
    return None


@app.websocket("/ws/liveness")
async def liveness_ws(websocket: WebSocket):
    await websocket.accept()
    ensure_model()

    session_id = str(uuid.uuid4())
    session = LivenessSession()
    session_start = time.time()

    # Tell the client their session ID immediately
    await websocket.send_text(json.dumps({
        "session_id": session_id,
        "state": "CALIBRATE",
        "prompt": "Hold still, calibrating..."
    }))

    with build_landmarker() as landmarker:
        try:
            while True:
                if time.time() - session_start > SESSION_TIMEOUT:
                    await websocket.send_text(json.dumps({
                        "session_id": session_id,
                        "state": "EXPIRED",
                        "prompt": "Session timed out.",
                        "result": None
                    }))
                    break
                raw = await websocket.receive_text()
                message = json.loads(raw)
                frame = decode_frame(message["frame"])

                timestamp_ms = int(time.time() * 1000)
                landmarks = run_mediapipe(landmarker, frame, timestamp_ms)
                status = session.process(landmarks)

                # Always include session_id in every response
                response = {"session_id": session_id, **status}
                await websocket.send_text(json.dumps(response))

                if status["state"] == "DONE" and status["result"] is not None:
                    # Persist the result before closing
                    session_store[session_id] = {
                        "session_id": session_id,
                        "result": status["result"],
                        "completed_at": time.time()
                    }
                    break

        except WebSocketDisconnect:
            pass

    await websocket.close()


@app.get("/api/liveness/result/{session_id}")
async def get_liveness_result(session_id: str):
    record = session_store.get(session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found or not yet complete")
    return record