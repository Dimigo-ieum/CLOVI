# client.py
import asyncio
import cv2
import websockets

SERVER_WS = "ws://127.0.0.1:8765"  # <-- change this

async def send_stream():
    cap = cv2.VideoCapture(0)  # use correct index/backend for your cam
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Optional: reduce resolution for bandwidth/CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    async with websockets.connect(SERVER_WS, max_size=2**23) as ws:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            await ws.send(buf.tobytes())

    cap.release()

if __name__ == "__main__":
    asyncio.run(send_stream())
