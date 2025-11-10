# server.py
import io
from typing import Optional, Tuple, List
from flask import Flask, request, jsonify, abort
import cv2, numpy as np
import mediapipe as mp

app = Flask(__name__)

# ---- MediaPipe Pose (경량 설정) ----
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, enable_segmentation=False, smooth_landmarks=True)

def _bbox_from_landmarks(image_shape, lm, idxs: List[int], pad=0.12) -> Optional[Tuple[int,int,int,int]]:
    H, W = image_shape[:2]
    xs, ys = [], []
    for i in idxs:
        p = lm[i]
        if p.visibility < 0.5:
            continue
        xs.append(p.x * W); ys.append(p.y * H)
    if not xs or not ys:
        return None
    x1, x2 = max(0, min(xs)), min(W, max(xs))
    y1, y2 = max(0, min(ys)), min(H, max(ys))
    pw, ph = (x2-x1)*pad, (y2-y1)*pad
    x1, x2 = int(max(0, x1-pw)), int(min(W, x2+pw))
    y1, y2 = int(max(0, y1-ph)), int(min(H, y2+ph))
    if (x2-x1) < 5 or (y2-y1) < 5:
        return None
    return x1, y1, x2, y2

def _get_upper_lower(frame_bgr):
    res = pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not res or not res.pose_landmarks:
        return None, None
    lm = res.pose_landmarks.landmark
    up = _bbox_from_landmarks(
        frame_bgr.shape, lm,
        [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
         mp_pose.PoseLandmark.LEFT_HIP.value,      mp_pose.PoseLandmark.RIGHT_HIP.value],
        pad=0.15
    )
    lo = _bbox_from_landmarks(
        frame_bgr.shape, lm,
        [mp_pose.PoseLandmark.LEFT_HIP.value,      mp_pose.PoseLandmark.RIGHT_HIP.value,
         mp_pose.PoseLandmark.LEFT_ANKLE.value,    mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        pad=0.12
    )
    return up, lo

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer")
def infer():
    if "frame" not in request.files:
        abort(400, description="missing multipart field 'frame'")
    file = request.files["frame"]

    # 간단한 타입 체크(선택)
    if file.mimetype not in ("image/png", "image/jpeg"):
        abort(415, description="only image/png or image/jpeg accepted")

    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        abort(400, description="invalid image data")

    up_box, lo_box = _get_upper_lower(img)

    result = {"type": "inference", "upper": None, "lower": None}
    if up_box:
        x1,y1,x2,y2 = up_box
        result["upper"] = {
            "type": "upper-garment",
            "color": {"name":"unknown","hex":"#000000"},
            "pattern": "unknown",
            "bbox": [x1,y1,x2,y2]
        }
    if lo_box:
        x1,y1,x2,y2 = lo_box
        result["lower"] = {
            "type": "lower-garment",
            "color": {"name":"unknown","hex":"#000000"},
            "pattern": "unknown",
            "bbox": [x1,y1,x2,y2]
        }
    return jsonify(result)

if __name__ == "__main__":
    # 개발용 실행 (운영은 gunicorn/waitress 권장)
    app.run(host="0.0.0.0", port=8765, threaded=True)

