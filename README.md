# Liveness Detection

A modular liveness detection microservice built with MediaPipe and FastAPI. Designed as a standalone component for eKYC and digital onboarding pipelines.

The service runs a challenge-response flow (blink, turn left, turn right) in a randomized order, scores the session based on completion and speed, and returns a structured result. It exposes a WebSocket API so any client with a camera can stream frames and receive real-time feedback.

## Project structure

```
liveness-check/
├── app/
│   ├── __init__.py
│   ├── session.py        core session logic, challenge state machine
│   ├── detector.py       webcam demo entry point
│   ├── blink.py          EAR-based blink detection
│   ├── head_pose.py      yaw ratio head turn detection
│   ├── scorer.py         liveness scoring
│   └── api.py            FastAPI WebSocket server
├── tests/
│   └── test_client.py    headless test client
├── demo.py               run the desktop webcam demo
└── requirements.txt
```

## Requirements

Python 3.10+ with the packages:
- opencv-python
- mediapipe
- fastapi
- uvicorn
- numpy
- websockets

And a webcam as well, obviously.

## Setup

```bash
git clone https://github.com/eko-071/liveness-check
cd liveness-check

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

The MediaPipe face landmarker model (around 30MB) downloads automatically on first run.

## Running

### Desktop demo

Runs the full challenge flow locally with a webcam window and on-screen overlay.

```bash
python demo.py
```

Controls: `R` to restart with a new challenge order, `Q` to quit.

### API server

```bash
uvicorn app.api:app --reload
```

The WebSocket endpoint is at `ws://localhost:8000/ws/liveness`.

### Test client

With the server running, open a second terminal:

```bash
python tests/test_client.py
```

Streams your webcam to the server headlessly and prints the session state and final result in the terminal.

## API reference

### `WS /ws/liveness`

One connection per session. Client sends one frame per message, server responds after every frame.

**Client sends:**

```json
{ "frame": "<base64 encoded JPEG>" }
```

**Server responds:**

```json
{
  "state": "BLINK",
  "prompt": "Please blink",
  "order": ["BLINK", "TURN_LEFT", "TURN_RIGHT"],
  "index": 0,
  "time_left": 5.2,
  "checks": {
    "BLINK":      { "passed": false, "timed_out": false, "elapsed": null },
    "TURN_LEFT":  { "passed": false, "timed_out": false, "elapsed": null },
    "TURN_RIGHT": { "passed": false, "timed_out": false, "elapsed": null }
  },
  "result": null
}
```

When `state` is `DONE`, `result` is populated and the server closes the connection:

```json
{
  "status": "live",
  "score": 0.91,
  "breakdown": {
    "weighted_score": 0.87,
    "consistency_bonus": 0.05,
    "timeout_penalty": 0.0
  },
  "checks": {
    "BLINK": true,
    "TURN_LEFT": true,
    "TURN_RIGHT": true
  }
}
```

## Challenges

Each session runs three challenges in a randomized order, each with a 7 second timeout. Failing a challenge does not end the session.

| Challenge | What is detected |
|-----------|-----------------|
| BLINK | EAR drops below calibrated threshold for 2+ consecutive frames |
| TURN_LEFT | Nose-to-eye ratio deviates left of calibrated baseline |
| TURN_RIGHT | Nose-to-eye ratio deviates right, after returning to center first |

## Scoring

Each passed challenge is scored by speed: `1.0 - (elapsed / timeout) * 0.4`, giving a range of `0.6` (slow) to `1.0` (fast). Failed challenges score `0.0`.

```
final = BLINK * 0.35 + TURN_LEFT * 0.30 + TURN_RIGHT * 0.30
      + consistency_bonus (max 0.05)
      - timeout_penalty (0.05 per timeout)
```

Status is `live` if final score `>= 0.60`, otherwise `spoof`.

## Calibration

The first 30 frames establish per-user baselines before any challenge starts:

- **EAR baseline:** average eye aspect ratio when open, blink threshold set at 75%
- **Yaw baseline:** nose-to-eye ratio when facing forward, turns detected relative to this

## Integration

```python
import asyncio, base64, json, cv2
import websockets

async def verify(frame_generator):
    async with websockets.connect("ws://localhost:8000/ws/liveness") as ws:
        async for frame in frame_generator:
            _, buf = cv2.imencode(".jpg", frame)
            payload = json.dumps({"frame": base64.b64encode(buf).decode()})
            await ws.send(payload)

            response = json.loads(await ws.recv())
            if response["state"] == "DONE":
                return response["result"]
```