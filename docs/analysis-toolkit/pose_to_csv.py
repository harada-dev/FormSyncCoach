"""動画 -> MediaPipe Pose ランドマーク CSV.

MediaPipe 1.x の Tasks API を使う。旧 `mp.solutions.pose` は 1.0 で削除済み。

出力 CSV は 1 行 = 1 フレームのワイド形式:
    frame, t_sec, <name>_wx, <name>_wy, <name>_wz, <name>_vis, <name>_px, <name>_py
  *_w? : pose_world_landmarks。単位メートル、原点は左右股関節の中点(骨盤中心)。
         → 関節角度の計算に使う。
  *_p? : pose_landmarks を画素に直したもの。
         → 絶対速度の計算に使う(world は骨盤固定なので並進が消えている)。

使い方:
    python pose_to_csv.py kick.mp4 -o kick_landmarks.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def ensure_model(kind: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pose_landmarker_{kind}.task"
    if not path.exists():
        print(f"[model] downloading {kind} ...", file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URLS[kind], path)
    return path


def build_header() -> list[str]:
    cols = ["frame", "t_sec"]
    for name in LANDMARK_NAMES:
        cols += [f"{name}_wx", f"{name}_wy", f"{name}_wz",
                 f"{name}_vis", f"{name}_px", f"{name}_py"]
    return cols


def run(video: Path, out_csv: Path, model_kind: str, cache_dir: Path) -> dict:
    model_path = ensure_model(model_kind, cache_dir)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"動画を開けません: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    n_frames = 0
    n_detected = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker, \
            out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(build_header())

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(n_frames / fps * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            row: list = [n_frames, n_frames / fps]
            if result.pose_world_landmarks and result.pose_landmarks:
                n_detected += 1
                world = result.pose_world_landmarks[0]
                image = result.pose_landmarks[0]
                for w, p in zip(world, image):
                    row += [w.x, w.y, w.z, w.visibility,
                            p.x * width, p.y * height]
            else:
                row += [""] * (len(LANDMARK_NAMES) * 6)
            writer.writerow(row)
            n_frames += 1

    cap.release()
    return {"frames": n_frames, "detected": n_detected,
            "fps": fps, "width": width, "height": height}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("landmarks.csv"))
    ap.add_argument("--model", choices=list(MODEL_URLS), default="heavy",
                    help="heavy が最も正確。lite は速いが下肢の精度が落ちる")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "mediapipe")
    args = ap.parse_args()

    info = run(args.video, args.out, args.model, args.cache)
    print(f"{args.out}: {info['detected']}/{info['frames']} frames detected, "
          f"{info['fps']:.1f} fps, {info['width']}x{info['height']}")
    if info["fps"] < 120:
        print(f"[warning] {info['fps']:.0f} fps ではインパクト(接触 約9 ms)を"
              f"分解できません。240 fps 以上を推奨。", file=sys.stderr)


if __name__ == "__main__":
    main()
