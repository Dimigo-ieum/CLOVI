# client.py
import time, sys
import requests
import cv2

SERVER = "http://127.0.0.1:8765/infer"  # 서버 IP로 교체
INTERVAL_SEC = 0.5

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라 열기 실패", file=sys.stderr); sys.exit(1)
    # Pi 3B 권장 저해상도
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    session = requests.Session()
    try:
        while True:
            ok, frame = cap.read()
            if not ok: 
                time.sleep(INTERVAL_SEC); 
                continue

            # PNG 인코드
            ok, buf = cv2.imencode(".png", frame)
            if not ok:
                time.sleep(INTERVAL_SEC); 
                continue

            files = {"frame": ("frame.png", buf.tobytes(), "image/png")}
            try:
                r = session.post(SERVER, files=files, timeout=5)
                if r.ok:
                    result = r.json()
                    print(result)  # 필요 시 파싱/표시
                else:
                    print("HTTP", r.status_code, r.text[:120], file=sys.stderr)
            except requests.RequestException as e:
                print("REQ ERR:", e, file=sys.stderr)
            time.sleep(INTERVAL_SEC)
    finally:
        cap.release()

if __name__ == "__main__":
    main()

