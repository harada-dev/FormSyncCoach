"""ランドマーク CSV -> キック動作の関節角度・角速度(線形代数).

計算の中身:
  1. 骨盤ローカル座標系   Gram-Schmidt 直交化で回転行列 R を作り p_local = R^T (p - o)
  2. 関節角度             theta = atan2(|a x b|, a . b)    (arccos より数値的に安定)
  3. 符号付き矢状面角度   theta = atan2((a x b) . n, a . b)  n = 骨盤左右軸
  4. セグメント角速度     omega = u x du/dt                (u は単位ベクトル)
  5. 微分                 Savitzky-Golay (多項式当てはめの解析微分)

使い方:
    python kick_metrics.py landmarks.csv --leg right -o metrics.csv
    python kick_metrics.py --selftest        # 既知形状で線形代数を検証
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

# ---------------------------------------------------------------- 線形代数


def unit(v: np.ndarray) -> np.ndarray:
    """行ごとに正規化。ゼロ長は NaN にする。"""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1e-12, v / n, np.nan)


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2 ベクトルのなす角 [deg]。atan2 形式なので 0 度・180 度付近でも壊れない。"""
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.einsum("...i,...i->...", a, b)
    return np.degrees(np.arctan2(cross, dot))


def signed_angle(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """法線 n まわりの符号付き角度 [deg]。屈曲/伸展の向きを区別したいとき。"""
    cross = np.cross(a, b)
    s = np.einsum("...i,...i->...", cross, unit(n))
    c = np.einsum("...i,...i->...", a, b)
    return np.degrees(np.arctan2(s, c))


def pelvis_frame(l_hip, r_hip, l_sh, r_sh):
    """骨盤基準の正規直交基底を作る。

    x = 右向き(左股関節 -> 右股関節)
    z = 上向き(肩中点 - 股関節中点 を x に直交化)
    y = z x x = 前向き
    戻り値 (origin, R)。R の列が各軸なので p_local = (p - origin) @ R。
    """
    origin = 0.5 * (l_hip + r_hip)
    x = unit(r_hip - l_hip)
    up = 0.5 * (l_sh + r_sh) - origin
    z = unit(up - np.einsum("...i,...i->...", up, x)[..., None] * x)  # Gram-Schmidt
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=-1)
    return origin, R


def to_local(p: np.ndarray, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.einsum("...ji,...j->...i", R, p - origin)


def sg_derivative(x: np.ndarray, dt: float, window: int = 9, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay による 1 階微分。生の差分よりノイズ増幅が小さい。"""
    window = min(window if window % 2 else window + 1, len(x) - (1 - len(x) % 2))
    if window < poly + 2:
        return np.gradient(x, dt, axis=0)
    return savgol_filter(x, window, poly, deriv=1, delta=dt, axis=0, mode="interp")


def segment_angular_velocity(u: np.ndarray, dt: float, **kw) -> np.ndarray:
    """単位ベクトル u(t) が表すセグメントの角速度 [rad/s]。

    omega = u x du/dt。2 点から作った軸なので、軸まわりの自転成分は
    原理的に観測できない(MediaPipe では下腿の内外旋は取れない)。
    """
    du = sg_derivative(u, dt, **kw)
    return np.cross(u, du)


# ---------------------------------------------------------------- CSV 読み込み


def load(csv_path: Path):
    raw = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    cols = {name: raw[name] for name in raw.dtype.names}
    return cols


def world(cols, name: str) -> np.ndarray:
    return np.stack([cols[f"{name}_wx"], cols[f"{name}_wy"], cols[f"{name}_wz"]], axis=-1)


def pixels(cols, name: str) -> np.ndarray:
    return np.stack([cols[f"{name}_px"], cols[f"{name}_py"]], axis=-1)


# ---------------------------------------------------------------- 指標


def compute(cols, leg: str, dt: float, m_per_px: float | None):
    other = "left" if leg == "right" else "right"

    l_hip, r_hip = world(cols, "left_hip"), world(cols, "right_hip")
    l_sh, r_sh = world(cols, "left_shoulder"), world(cols, "right_shoulder")
    origin, R = pelvis_frame(l_hip, r_hip, l_sh, r_sh)

    def local(name):
        return to_local(world(cols, name), origin, R)

    hip = local(f"{leg}_hip")
    knee = local(f"{leg}_knee")
    ankle = local(f"{leg}_ankle")
    toe = local(f"{leg}_foot_index")
    sup_hip = local(f"{other}_hip")
    sup_knee = local(f"{other}_knee")
    sup_ankle = local(f"{other}_ankle")

    thigh = unit(knee - hip)
    shank = unit(ankle - knee)
    foot = unit(toe - ankle)
    lateral = np.tile(np.array([1.0, 0.0, 0.0]), (len(hip), 1))  # ローカル系の右向き軸

    out = {}
    # 膝: 2 つの慣習を両方出す。論文によって定義が違うため。
    out["knee_flexion_deg"] = angle_between(thigh, shank)          # 0 = 完全伸展
    out["knee_included_deg"] = 180.0 - out["knee_flexion_deg"]     # 180 = 完全伸展
    # 足首: 距腿関節そのものではなく「下腿 - 足部長軸」の近似角。
    out["ankle_shank_foot_deg"] = angle_between(shank, foot)
    # 股関節: 矢状面の符号付き(+ 屈曲 / - 伸展)
    trunk = unit(0.5 * (l_sh + r_sh) - 0.5 * (l_hip + r_hip))
    out["hip_sagittal_deg"] = signed_angle(to_local(trunk, 0.0, R), thigh, lateral)
    out["support_knee_flexion_deg"] = angle_between(
        unit(sup_knee - sup_hip), unit(sup_ankle - sup_knee))

    # 角速度
    out["knee_ext_vel_dps"] = -sg_derivative(out["knee_flexion_deg"], dt)
    out["shank_ang_vel_dps"] = np.degrees(
        np.linalg.norm(segment_angular_velocity(shank, dt), axis=-1))
    out["thigh_ang_vel_dps"] = np.degrees(
        np.linalg.norm(segment_angular_velocity(thigh, dt), axis=-1))

    # 骨盤 - 肩の分離角(水平面)
    hip_axis = unit(r_hip - l_hip)[:, [0, 2]]
    sh_axis = unit(r_sh - l_sh)[:, [0, 2]]
    out["pelvis_shoulder_sep_deg"] = np.degrees(np.arctan2(
        hip_axis[:, 0] * sh_axis[:, 1] - hip_axis[:, 1] * sh_axis[:, 0],
        np.einsum("ij,ij->i", hip_axis, sh_axis)))

    # 足部速度。world は骨盤原点なので絶対速度にならない -> 画素系から求める。
    foot_com_w = 0.5 * (world(cols, f"{leg}_ankle") + world(cols, f"{leg}_foot_index"))
    out["foot_speed_rel_ms"] = np.linalg.norm(sg_derivative(foot_com_w, dt), axis=-1)
    if m_per_px is not None:
        foot_com_p = 0.5 * (pixels(cols, f"{leg}_ankle") + pixels(cols, f"{leg}_foot_index"))
        v_px = sg_derivative(foot_com_p, dt)
        out["foot_speed_abs_ms"] = np.linalg.norm(v_px, axis=-1) * m_per_px
    return out


def detect_contact(out: dict) -> int | None:
    """足部速度がピークをとった直後の最大減速フレームを接触候補とする。"""
    key = "foot_speed_abs_ms" if "foot_speed_abs_ms" in out else "foot_speed_rel_ms"
    v = out[key]
    if np.all(np.isnan(v)):
        return None
    peak = int(np.nanargmax(v))
    tail = v[peak:]
    if len(tail) < 3:
        return peak
    return peak + int(np.nanargmin(np.gradient(tail)))


# ---------------------------------------------------------------- 自己検証


def selftest() -> int:
    """既知の幾何を与えて角度・角速度が復元できるか確かめる。"""
    n, dt = 200, 1 / 240
    t = np.arange(n) * dt
    truth = 90.0 - 60.0 * t / t[-1]          # 膝屈曲 90 度 -> 30 度の直線変化
    rad = np.radians(truth)

    hip = np.zeros((n, 3))
    knee = np.tile([0.0, 0.0, -0.45], (n, 1))            # 大腿 45 cm、真下
    shank_dir = np.stack([np.zeros(n), np.sin(rad), -np.cos(rad)], axis=-1)
    ankle = knee + 0.42 * shank_dir

    thigh = unit(knee - hip)
    shank = unit(ankle - knee)
    got = angle_between(thigh, shank)
    err_angle = np.max(np.abs(got - truth))

    d_truth = np.full(n, (truth[-1] - truth[0]) / (t[-1] - t[0]))
    d_got = sg_derivative(got, dt)
    err_rate = np.max(np.abs(d_got - d_truth))

    omega = np.degrees(np.linalg.norm(segment_angular_velocity(shank, dt), axis=-1))
    err_omega = np.max(np.abs(omega[5:-5] - abs(d_truth[0])))

    # 骨盤ローカル系: 全身を任意に回転しても局所座標が不変であること
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(4, 3))
    l_hip, r_hip, l_sh, r_sh = pts
    theta = 0.7
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    a = to_local(*(x[None] for x in (pts[0],)), *pelvis_frame(*(x[None] for x in pts)))
    rot = pts @ Rz.T
    b = to_local(rot[0][None], *pelvis_frame(*(x[None] for x in rot)))
    err_frame = np.max(np.abs(a - b))

    print(f"角度   最大誤差 {err_angle:.3e} deg")
    print(f"角速度 最大誤差 {err_rate:.3e} deg/s  (真値 {d_truth[0]:.1f})")
    print(f"omega  最大誤差 {err_omega:.3e} deg/s")
    print(f"骨盤系 回転不変性 誤差 {err_frame:.3e} m")
    ok = err_angle < 1e-9 and err_rate < 1e-6 and err_omega < 1e-6 and err_frame < 1e-12
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", type=Path)
    ap.add_argument("--leg", choices=["right", "left"], default="right")
    ap.add_argument("--fps", type=float, default=None, help="未指定なら t_sec 列から推定")
    ap.add_argument("--m-per-px", type=float, default=None,
                    help="画素->メートル換算(例: ボール直径0.22m / 画素直径)")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.csv is None:
        ap.error("csv か --selftest が必要です")

    cols = load(args.csv)
    t = cols["t_sec"]
    dt = 1.0 / args.fps if args.fps else float(np.median(np.diff(t)))
    out = compute(cols, args.leg, dt, args.m_per_px)

    contact = detect_contact(out)
    print(f"fps = {1/dt:.1f},  frames = {len(t)}")
    if contact is not None:
        print(f"接触候補フレーム = {contact} (t = {t[contact]:.4f} s)")
        for k, v in out.items():
            print(f"  {k:28s} {v[contact]:10.2f}   (peak {np.nanmax(np.abs(v)):.2f})")

    if args.out:
        names = list(out)
        data = np.column_stack([cols["frame"], t] + [out[k] for k in names])
        np.savetxt(args.out, data, delimiter=",", fmt="%.6f",
                   header="frame,t_sec," + ",".join(names), comments="")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
