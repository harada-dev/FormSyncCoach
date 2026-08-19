"""検証用: 人物静止画からパン/ズームする短い動画を作る。"""
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

CANDIDATES = [
    "https://storage.googleapis.com/mediapipe-assets/pose.jpg",
    "https://storage.googleapis.com/mediapipe-assets/pose_segmentation.jpg",
    "https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Ronaldinho_kick.jpg/640px-Ronaldinho_kick.jpg",
]

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
img = None
for url in CANDIDATES:
    try:
        data = urllib.request.urlopen(url, timeout=20).read()
        arr = np.frombuffer(data, np.uint8)
        cand = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if cand is not None:
            img = cand
            print("using", url, cand.shape)
            break
    except Exception as e:
        print("skip", url, type(e).__name__, e)

if img is None:
    raise SystemExit("人物画像を取得できませんでした")

h, w = img.shape[:2]
scale = 720 / max(h, w)
img = cv2.resize(img, (int(w * scale), int(h * scale)))
h, w = img.shape[:2]

fps, n = 240, 96
vw = cv2.VideoWriter(str(out_dir / "test.mp4"),
                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i in range(n):
    dx = int(24 * np.sin(2 * np.pi * i / n))
    dy = int(10 * np.sin(4 * np.pi * i / n))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    vw.write(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
vw.release()
print("wrote", out_dir / "test.mp4", f"{w}x{h} {fps}fps {n}frames")
