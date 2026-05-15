import asyncio
import base64
import json
import cv2
import websockets


SERVER = "ws://localhost:8000/ws/liveness"


def encode_frame(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


async def run():
    cap = cv2.VideoCapture(0)
    print(f"Connecting to {SERVER}")

    async with websockets.connect(SERVER) as ws:
        print("Connected. Starting session...\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            payload = json.dumps({"frame": encode_frame(frame)})
            await ws.send(payload)

            response = json.loads(await ws.recv())
            state  = response["state"]
            prompt = response["prompt"]
            tl     = response["time_left"]
            result = response["result"]

            if prompt:
                print(f"[{state}] {prompt} ({tl:.1f}s remaining)", end="\r")

            if state == "DONE" and result:
                print(f"Done.")
                print(f"Result : {result['status'].upper()}")
                print(f"Score : {result['score']}")
                print(f"Breakdown : {result['breakdown']}")
                print(f"Checks : {result['checks']}")
                break

    cap.release()


if __name__ == "__main__":
    asyncio.run(run())