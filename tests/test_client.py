import asyncio
import base64
import json
import time
import numpy as np
import cv2
import websockets
import requests

SERVER_WS  = "ws://localhost:8000/ws/liveness"
SERVER_HTTP = "http://localhost:8000"


def blank_frame() -> str:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode("utf-8")


async def test_websocket_handshake():
    print("Test 1: WebSocket handshake and session ID")
    print()

    async with websockets.connect(SERVER_WS) as ws:
        # First message should contain session_id
        first = json.loads(await ws.recv())
        session_id = first.get("session_id")

        assert session_id is not None, "No session_id in first message"
        assert first.get("state") == "CALIBRATE", "Expected CALIBRATE state first"
        print(f"Session ID received : {session_id}")
        print(f"Initial state : {first['state']}")

        # Send a few blank frames and confirm server keeps responding
        for i in range(5):
            await ws.send(json.dumps({"frame": blank_frame()}))
            response = json.loads(await ws.recv())
            assert "session_id" in response, "session_id missing from frame response"
            assert "state" in response, "state missing from frame response"

        print(f"Server responded to 5 blank frames correctly")
        print(f"Test 1 PASSED")
        return session_id


def test_result_endpoint(session_id: str):
    print("Test 2: GET endpoint returns stored result")
    print()

    # Hit a non-existent session first — should 404
    r = requests.get(f"{SERVER_HTTP}/api/liveness/result/nonexistent-id")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("Non-existent session correctly returns 404")
    print(f"Test 2 PASSED\n")


async def main():
    print("API Tests")
    try:
        session_id = await test_websocket_handshake()
        test_result_endpoint(session_id)
        print("All tests passed.")
        print(f"To test a real result, complete a full liveness session")
        print(f"and curl: {SERVER_HTTP}/api/liveness/result/<session_id>")
    except AssertionError as e:
        print(f"FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())