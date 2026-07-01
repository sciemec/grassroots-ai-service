"""
GrassRoots Sports — AI Tracking Service v3
YOLOv8x + MediaPipe BlazePose + supervision ByteTracker

POST /track           — match video → per-player positions, heatmaps, ball tracking, speed
POST /athletic-test   — drill video → jump/sprint/balance/endurance scoring (MediaPipe)
POST /analyse-drill   — drill video → full biomechanical scorecard (MediaPipe)
POST /analyse         — match video → Gemini 2.0 Flash + Claude tactical analysis
POST /clip            — match video → sprint highlight clips → R2
POST /process-video   — background pipeline (download R2 → track → clip → callback)
POST /gemini-upload   — proxy video upload to Gemini File API (CORS bypass)
POST /generate-thumbnail — extract frame at 3s → upload JPEG to R2
GET  /job/{job_id}    — poll background analysis job status
GET  /health          — liveness check
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import subprocess
import tempfile
import time
import urllib.request
import uuid as uuid_mod
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Optional

import boto3
import cv2
import httpx
import mediapipe as mp
import numpy as np
import supervision as sv
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pydantic import BaseModel
from sklearn.cluster import KMeans
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="GrassRoots AI Tracker", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://grassrootssports.live",
        "https://www.grassrootssports.live",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup — download MediaPipe model once
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def download_pose_model() -> None:
    model_path = "pose_landmarker_heavy.task"
    if not os.path.exists(model_path):
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_heavy/float16/latest/"
            "pose_landmarker_heavy.task"
        )
        urllib.request.urlretrieve(url, model_path)

# ---------------------------------------------------------------------------
# YOLOv8 — lazy loaded
# ---------------------------------------------------------------------------

_model: YOLO | None = None
_jobs: dict[str, dict] = {}


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8x.pt")
    return _model


# ---------------------------------------------------------------------------
# MediaPipe Pose Landmarker — lazy loaded
# ---------------------------------------------------------------------------

_pose_landmarker: mp_vision.PoseLandmarker | None = None


def get_pose_landmarker() -> mp_vision.PoseLandmarker:
    global _pose_landmarker
    if _pose_landmarker is None:
        base_options = mp_python.BaseOptions(
            model_asset_path="pose_landmarker_heavy.task"
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PITCH_LENGTH_M  = 105.0
PITCH_WIDTH_M   = 68.0
HEATMAP_COLS    = 20
HEATMAP_ROWS    = 13
SAMPLE_FPS      = 1
PERSON_CLASS_ID = 0
BALL_CLASS_ID   = 32
SPRINT_THRESHOLD_KMH = 25.0

TRACKER_CONFIG = SimpleNamespace(
    track_activation_threshold=0.25,
    lost_track_buffer=50,
    minimum_matching_threshold=0.8,
    frame_rate=SAMPLE_FPS,
    minimum_consecutive_frames=3,
)

# ---------------------------------------------------------------------------
# MediaPipe landmark indices
# ---------------------------------------------------------------------------

class LM:
    NOSE         = 0
    L_SHOULDER   = 11;  R_SHOULDER  = 12
    L_ELBOW      = 13;  R_ELBOW     = 14
    L_WRIST      = 15;  R_WRIST     = 16
    L_HIP        = 23;  R_HIP       = 24
    L_KNEE       = 25;  R_KNEE      = 26
    L_ANKLE      = 27;  R_ANKLE     = 28
    L_FOOT_INDEX = 31;  R_FOOT_INDEX= 32

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def angle_3d(a: Any, b: Any, c: Any) -> float:
    ax, ay, az = a.x - b.x, a.y - b.y, a.z - b.z
    cx, cy, cz = c.x - b.x, c.y - b.y, c.z - b.z
    dot   = ax*cx + ay*cy + az*cz
    mag_a = math.sqrt(ax**2 + ay**2 + az**2)
    mag_c = math.sqrt(cx**2 + cy**2 + cz**2)
    if mag_a == 0 or mag_c == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_a * mag_c)))))

def midpoint(a: Any, b: Any) -> Any:
    class P:
        x = (a.x + b.x) / 2
        y = (a.y + b.y) / 2
        z = (a.z + b.z) / 2
    return P()

def visible(lm: list, *indices: int, threshold: float = 0.45) -> bool:
    return all(getattr(lm[i], "visibility", 1.0) >= threshold for i in indices)

# ---------------------------------------------------------------------------
# Per-frame biomechanical measurement
# ---------------------------------------------------------------------------

def measure_frame(world_lm: list) -> dict[str, float]:
    m: dict[str, float] = {}
    if visible(world_lm, LM.L_HIP, LM.L_KNEE, LM.L_ANKLE):
        m["left_knee_flexion"]  = angle_3d(world_lm[LM.L_HIP],  world_lm[LM.L_KNEE],  world_lm[LM.L_ANKLE])
    if visible(world_lm, LM.R_HIP, LM.R_KNEE, LM.R_ANKLE):
        m["right_knee_flexion"] = angle_3d(world_lm[LM.R_HIP],  world_lm[LM.R_KNEE],  world_lm[LM.R_ANKLE])
    if visible(world_lm, LM.L_SHOULDER, LM.L_HIP, LM.L_KNEE):
        m["left_hip_flexion"]   = angle_3d(world_lm[LM.L_SHOULDER], world_lm[LM.L_HIP], world_lm[LM.L_KNEE])
    if visible(world_lm, LM.R_SHOULDER, LM.R_HIP, LM.R_KNEE):
        m["right_hip_flexion"]  = angle_3d(world_lm[LM.R_SHOULDER], world_lm[LM.R_HIP], world_lm[LM.R_KNEE])
    if visible(world_lm, LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST):
        m["left_elbow_flexion"]  = angle_3d(world_lm[LM.L_SHOULDER], world_lm[LM.L_ELBOW], world_lm[LM.L_WRIST])
    if visible(world_lm, LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST):
        m["right_elbow_flexion"] = angle_3d(world_lm[LM.R_SHOULDER], world_lm[LM.R_ELBOW], world_lm[LM.R_WRIST])
    m["com_height"]  = (world_lm[LM.L_HIP].y + world_lm[LM.R_HIP].y) / 2
    m["hip_drop_mm"] = abs(world_lm[LM.L_HIP].y - world_lm[LM.R_HIP].y) * 1000
    hip_mid      = midpoint(world_lm[LM.L_HIP],     world_lm[LM.R_HIP])
    shoulder_mid = midpoint(world_lm[LM.L_SHOULDER], world_lm[LM.R_SHOULDER])
    trunk_dx = shoulder_mid.x - hip_mid.x
    trunk_dy = shoulder_mid.y - hip_mid.y
    m["trunk_lean_deg"] = abs(math.degrees(math.atan2(abs(trunk_dx), max(abs(trunk_dy), 0.001))))
    if visible(world_lm, LM.L_KNEE, LM.L_ANKLE):
        m["left_knee_valgus_mm"]  = (world_lm[LM.L_ANKLE].x - world_lm[LM.L_KNEE].x) * 1000
    if visible(world_lm, LM.R_KNEE, LM.R_ANKLE):
        m["right_knee_valgus_mm"] = (world_lm[LM.R_KNEE].x  - world_lm[LM.R_ANKLE].x) * 1000
    return m

# ---------------------------------------------------------------------------
# Aggregate frame measurements
# ---------------------------------------------------------------------------

def aggregate(frames: list[dict]) -> dict[str, dict]:
    if not frames:
        return {}
    keys = set(k for f in frames for k in f)
    result = {}
    half = len(frames) // 2
    for k in keys:
        vals = [f[k] for f in frames if k in f]
        if not vals:
            continue
        result[k] = {
            "mean":            round(sum(vals) / len(vals), 2),
            "min":             round(min(vals), 2),
            "max":             round(max(vals), 2),
            "first_half_mean": round(sum(vals[:half]) / max(len(vals[:half]), 1), 2),
            "last_half_mean":  round(sum(vals[half:]) / max(len(vals[half:]), 1), 2),
        }
    return result

# ---------------------------------------------------------------------------
# Shared MediaPipe video processor
# ---------------------------------------------------------------------------

def process_video_with_mediapipe(
    video_path: str,
    target_fps: int = 10,
) -> tuple[list[dict], dict]:
    landmarker   = get_pose_landmarker()
    cap          = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Cannot open video file")
    orig_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_every = max(1, int(round(orig_fps / target_fps)))
    frame_measurements: list[dict] = []
    frame_idx    = 0
    timestamp_ms = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, int(timestamp_ms))
            if result.pose_world_landmarks:
                m = measure_frame(result.pose_world_landmarks[0])
                if m:
                    frame_measurements.append(m)
            timestamp_ms += 1000.0 / target_fps
        frame_idx += 1
    cap.release()
    if len(frame_measurements) < 5:
        raise HTTPException(
            status_code=422,
            detail="Not enough pose detections. Ensure your full body is visible and lighting is good.",
        )
    return frame_measurements, aggregate(frame_measurements)

# ---------------------------------------------------------------------------
# Age-group benchmarks
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, dict] = {
    "u13": {
        "jump_com_drop":         {"elite": 0.18, "good": 0.14, "needs_work": 0.10},
        "jump_landing_knee":     {"elite": 105,  "good": 120,  "needs_work": 135},
        "sprint_trunk_lean":     {"elite": 18,   "good": 14,   "needs_work": 10},
        "sprint_asymmetry":      {"elite": 6,    "good": 10,   "needs_work": 15},
        "balance_hip_drop":      {"elite": 8,    "good": 18,   "needs_work": 30},
        "balance_knee_valgus":   {"elite": 8,    "good": 18,   "needs_work": 30},
        "endurance_degradation": {"elite": 5,    "good": 12,   "needs_work": 20},
    },
    "u17": {
        "jump_com_drop":         {"elite": 0.22, "good": 0.17, "needs_work": 0.12},
        "jump_landing_knee":     {"elite": 100,  "good": 115,  "needs_work": 130},
        "sprint_trunk_lean":     {"elite": 22,   "good": 17,   "needs_work": 12},
        "sprint_asymmetry":      {"elite": 5,    "good": 8,    "needs_work": 12},
        "balance_hip_drop":      {"elite": 6,    "good": 14,   "needs_work": 25},
        "balance_knee_valgus":   {"elite": 6,    "good": 14,   "needs_work": 25},
        "endurance_degradation": {"elite": 4,    "good": 10,   "needs_work": 18},
    },
    "senior": {
        "jump_com_drop":         {"elite": 0.25, "good": 0.20, "needs_work": 0.14},
        "jump_landing_knee":     {"elite": 95,   "good": 110,  "needs_work": 125},
        "sprint_trunk_lean":     {"elite": 26,   "good": 20,   "needs_work": 14},
        "sprint_asymmetry":      {"elite": 4,    "good": 7,    "needs_work": 10},
        "balance_hip_drop":      {"elite": 5,    "good": 12,   "needs_work": 22},
        "balance_knee_valgus":   {"elite": 5,    "good": 12,   "needs_work": 22},
        "endurance_degradation": {"elite": 3,    "good": 8,    "needs_work": 15},
    },
}

def percentile_from_bench(value: float, bench: dict, higher_is_better: bool = True) -> int:
    elite = bench["elite"]; good = bench["good"]; needs = bench["needs_work"]
    if higher_is_better:
        if value >= elite: return 95
        if value >= good:  return 95 - int((elite - value) / max(elite - good,  0.001) * 30)
        if value >= needs: return 65 - int((good  - value) / max(good  - needs, 0.001) * 35)
        return max(5, 30 - int((needs - value) / max(needs, 0.001) * 25))
    else:
        if value <= elite: return 95
        if value <= good:  return 95 - int((value - elite) / max(good  - elite, 0.001) * 30)
        if value <= needs: return 65 - int((value - good)  / max(needs - good,  0.001) * 35)
        return max(5, 30 - int((value - needs) / max(needs, 0.001) * 25))

def rating_from_pct(pct: int) -> str:
    if pct >= 85: return "Elite"
    if pct >= 70: return "Advanced"
    if pct >= 55: return "Developing"
    if pct >= 40: return "Foundation"
    return "Beginner"

# ---------------------------------------------------------------------------
# Test-specific scorers
# ---------------------------------------------------------------------------

def score_jump(agg: dict, age: str) -> dict[str, Any]:
    bench     = BENCHMARKS[age]
    com_range = (agg["com_height"]["max"] - agg["com_height"]["min"]) if "com_height" in agg else 0.15
    jump_pct  = percentile_from_bench(com_range, bench["jump_com_drop"], higher_is_better=True)
    avg_knee  = None
    if "left_knee_flexion" in agg and "right_knee_flexion" in agg:
        avg_knee = (agg["left_knee_flexion"]["min"] + agg["right_knee_flexion"]["min"]) / 2
    land_pct  = percentile_from_bench(avg_knee or 120, bench["jump_landing_knee"], higher_is_better=False)
    overall   = round(jump_pct * 0.6 + land_pct * 0.4)
    return {
        "test_score": overall, "percentile": overall, "rating": rating_from_pct(overall),
        "key_metric": f"Power index: {round(com_range * 100)}cm COM drop",
        "detail": (
            f"Jump scored {overall}th percentile ({age}). "
            f"COM drop {round(com_range*100)}cm — {'excellent explosive power' if jump_pct >= 70 else 'work on hip flexor strength and plyometrics'}. "
            f"Landing: {'safe knee absorption' if land_pct >= 65 else 'stiff landing — increase knee bend to reduce injury risk'}."
        ),
        "injury_flag": land_pct < 40,
        "injury_detail": "Stiff landing detected — increase knee bend on landing." if land_pct < 40 else None,
    }

def score_sprint(agg: dict, age: str) -> dict[str, Any]:
    bench    = BENCHMARKS[age]
    trunk    = agg.get("trunk_lean_deg", {}).get("max", 15)
    lean_pct = percentile_from_bench(trunk, bench["sprint_trunk_lean"], higher_is_better=True)
    l_knee   = agg.get("left_knee_flexion",  {}).get("mean")
    r_knee   = agg.get("right_knee_flexion", {}).get("mean")
    asym_pct = 50
    if l_knee and r_knee and max(l_knee, r_knee) > 0:
        asym     = abs(l_knee - r_knee) / max(l_knee, r_knee) * 100
        asym_pct = percentile_from_bench(asym, bench["sprint_asymmetry"], higher_is_better=False)
    overall  = round(lean_pct * 0.55 + asym_pct * 0.45)
    return {
        "test_score": overall, "percentile": overall, "rating": rating_from_pct(overall),
        "key_metric": f"Trunk lean: {round(trunk, 1)}°",
        "detail": (
            f"Sprint mechanics scored {overall}th percentile. "
            f"Trunk lean {round(trunk, 1)}° — {'excellent forward drive' if lean_pct >= 70 else 'more forward lean will improve acceleration'}. "
            f"Stride symmetry: {'balanced' if asym_pct >= 70 else 'imbalanced — overuse injury risk on dominant side'}."
        ),
    }

def score_balance(agg: dict, age: str) -> dict[str, Any]:
    bench    = BENCHMARKS[age]
    hip_drop = agg.get("hip_drop_mm",          {}).get("mean", 20)
    drop_pct = percentile_from_bench(hip_drop, bench["balance_hip_drop"],    higher_is_better=False)
    l_val    = abs(agg.get("left_knee_valgus_mm",  {}).get("mean", 0))
    r_val    = abs(agg.get("right_knee_valgus_mm", {}).get("mean", 0))
    avg_val  = (l_val + r_val) / 2
    val_pct  = percentile_from_bench(avg_val, bench["balance_knee_valgus"], higher_is_better=False)
    overall  = round(drop_pct * 0.5 + val_pct * 0.5)
    return {
        "test_score": overall, "percentile": overall, "rating": rating_from_pct(overall),
        "key_metric": f"Hip drop: {round(hip_drop)}mm · Knee valgus: {round(avg_val)}mm",
        "detail": (
            f"Balance scored {overall}th percentile. "
            f"Hip drop {round(hip_drop)}mm — {'excellent glute stability' if drop_pct >= 70 else 'weak hip abductors — key ACL risk factor'}. "
            f"Knee alignment: {'safe and controlled' if val_pct >= 65 else 'valgus detected — strengthen hip abductors urgently'}."
        ),
        "injury_flag": val_pct < 45 or drop_pct < 40,
        "injury_detail": "Knee valgus and/or hip drop detected — ACL risk elevated." if (val_pct < 45 or drop_pct < 40) else None,
    }

def score_endurance(agg: dict, age: str) -> dict[str, Any]:
    bench = BENCHMARKS[age]
    degs  = []
    for key in ["left_knee_flexion", "right_knee_flexion", "trunk_lean_deg"]:
        d = agg.get(key, {})
        f, l = d.get("first_half_mean"), d.get("last_half_mean")
        if f and l and f > 0:
            degs.append(abs(l - f) / f * 100)
    avg_deg = sum(degs) / len(degs) if degs else 10
    pct     = percentile_from_bench(avg_deg, bench["endurance_degradation"], higher_is_better=False)
    return {
        "test_score": pct, "percentile": pct, "rating": rating_from_pct(pct),
        "key_metric": f"Technique degradation: {round(avg_deg, 1)}%",
        "detail": (
            f"Endurance scored {pct}th percentile. "
            f"Technique degraded {round(avg_deg, 1)}% from first to last reps. "
            f"{'Excellent fatigue resistance.' if pct >= 70 else 'Significant breakdown under fatigue — focus on cardiovascular conditioning.'}"
        ),
        "injury_flag": avg_deg > 20,
        "injury_detail": f"Technique degraded {round(avg_deg, 1)}% under fatigue — injury risk rises in late match stages." if avg_deg > 20 else None,
    }

def score_general(agg: dict, age: str) -> dict[str, Any]:
    l   = agg.get("left_knee_flexion",  {}).get("mean", 120)
    r   = agg.get("right_knee_flexion", {}).get("mean", 120)
    pct = percentile_from_bench((l + r) / 2, BENCHMARKS[age]["jump_landing_knee"], higher_is_better=False)
    return {
        "test_score": pct, "percentile": pct, "rating": rating_from_pct(pct),
        "key_metric": f"Avg knee flexion: {round((l+r)/2, 1)}°",
        "detail": f"General athletic movement scored {pct}th percentile.",
    }

# ---------------------------------------------------------------------------
# POST /athletic-test
# ---------------------------------------------------------------------------

@app.post("/athletic-test")
async def athletic_test(
    file: UploadFile = File(...),
    test_type: str = "general",
    age_group: str = "u17",
) -> dict[str, Any]:
    if age_group not in BENCHMARKS:
        age_group = "u17"
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    suffix = os.path.splitext(file.filename or "test.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
    try:
        _, agg = process_video_with_mediapipe(tmp_path)
        scorers = {"jump": score_jump, "sprinting": score_sprint, "balance": score_balance, "endurance": score_endurance}
        return scorers.get(test_type, score_general)(agg, age_group)
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

# ---------------------------------------------------------------------------
# POST /analyse-drill
# ---------------------------------------------------------------------------

@app.post("/analyse-drill")
async def analyse_drill(
    file: UploadFile = File(...),
    drill_type: str = "general",
    age_group:  str = "u17",
) -> dict[str, Any]:
    if age_group not in BENCHMARKS:
        age_group = "u17"
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    suffix = os.path.splitext(file.filename or "drill.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
    try:
        frame_data, agg = process_video_with_mediapipe(tmp_path)
        scores = []
        l_knee = agg.get("left_knee_flexion",  {}).get("mean")
        r_knee = agg.get("right_knee_flexion", {}).get("mean")
        if l_knee and r_knee:
            avg_k = (l_knee + r_knee) / 2
            kp    = percentile_from_bench(avg_k, BENCHMARKS[age_group]["jump_landing_knee"], higher_is_better=False)
            scores.append({"metric": "Knee Flexion", "value_deg": round(avg_k, 1), "percentile": kp, "rating": rating_from_pct(kp),
                "detail": f"Avg knee flexion {round(avg_k,1)}° — {'excellent absorption' if kp >= 70 else 'increase knee bend for better force absorption'}.",
                "what_it_means": "Knee flexion measures shock absorption. More bend = less injury risk and more explosive power."})
        trunk = agg.get("trunk_lean_deg", {}).get("mean")
        if trunk:
            tp = percentile_from_bench(trunk, BENCHMARKS[age_group]["sprint_trunk_lean"], higher_is_better=True)
            scores.append({"metric": "Trunk Lean", "value_deg": round(trunk, 1), "percentile": tp, "rating": rating_from_pct(tp),
                "detail": f"Trunk lean {round(trunk,1)}° — {'strong forward drive' if tp >= 65 else 'more lean will improve power and efficiency'}.",
                "what_it_means": "Trunk lean determines how effectively you direct force into the ground."})
        if l_knee and r_knee and max(l_knee, r_knee) > 0:
            asym = abs(l_knee - r_knee) / max(l_knee, r_knee) * 100
            sp   = percentile_from_bench(asym, BENCHMARKS[age_group]["sprint_asymmetry"], higher_is_better=False)
            scores.append({"metric": "Bilateral Symmetry", "value_asymmetry_pct": round(asym, 1), "percentile": sp, "rating": rating_from_pct(sp),
                "detail": f"L/R difference {round(asym,1)}% — {'excellent symmetry' if sp >= 70 else 'asymmetry increases overuse injury risk'}.",
                "what_it_means": "Symmetry above 10% difference means one side compensates for the other — predicts overuse injuries."})
        flags = []; risk = 0
        mv  = max(abs(agg.get("left_knee_valgus_mm",  {}).get("mean", 0)), abs(agg.get("right_knee_valgus_mm", {}).get("mean", 0)))
        vp  = percentile_from_bench(mv, BENCHMARKS[age_group]["balance_knee_valgus"], higher_is_better=False)
        if mv > BENCHMARKS[age_group]["balance_knee_valgus"]["good"]:
            risk += 25
            flags.append({"name": "Knee Valgus", "severity": "High" if mv > BENCHMARKS[age_group]["balance_knee_valgus"]["needs_work"] else "Moderate",
                "detected": True, "measurement": f"{round(mv)}mm", "percentile": vp,
                "detail": f"Knee collapsed {round(mv)}mm inward — #1 ACL predictor. Strengthen hip abductors immediately.",
                "drills_to_fix": ["Lateral band walks 3×15", "Clamshells 3×20", "Single-leg squat focus"], "timeline": "6-8 weeks"})
        else:
            flags.append({"name": "Knee Valgus", "severity": "None", "detected": False, "measurement": f"{round(mv)}mm", "percentile": vp, "detail": "Good knee alignment throughout."})
        degs = []
        for key in ["left_knee_flexion", "right_knee_flexion"]:
            d = agg.get(key, {})
            if d.get("first_half_mean") and d.get("last_half_mean") and d["first_half_mean"] > 0:
                degs.append(abs(d["last_half_mean"] - d["first_half_mean"]) / d["first_half_mean"] * 100)
        avg_d = sum(degs) / len(degs) if degs else 5
        fp    = percentile_from_bench(avg_d, BENCHMARKS[age_group]["endurance_degradation"], higher_is_better=False)
        if avg_d > BENCHMARKS[age_group]["endurance_degradation"]["good"]:
            risk += 15
            flags.append({"name": "Fatigue Breakdown", "severity": "Moderate", "detected": True, "measurement": f"{round(avg_d,1)}% degradation", "percentile": fp,
                "detail": f"Technique degraded {round(avg_d,1)}% — injury risk rises when fatigued.",
                "drills_to_fix": ["Interval training 3×/week", "Technical drills after cardio"], "timeline": "6-10 weeks"})
        else:
            flags.append({"name": "Fatigue Breakdown", "severity": "None", "detected": False, "measurement": f"{round(avg_d,1)}%", "percentile": fp, "detail": "Good technique maintenance throughout."})
        rl = "Critical" if risk >= 60 else "High" if risk >= 40 else "Moderate" if risk >= 20 else "Low" if risk >= 10 else "Excellent"
        op = round(sum(s["percentile"] for s in scores) / max(len(scores), 1))
        return {
            "overall_percentile": op, "overall_rating": rating_from_pct(op),
            "drill_type": drill_type, "drill_description": drill_type.replace("_", " ").title(),
            "age_group": age_group, "frames_analysed": len(frame_data),
            "scores": scores,
            "injury_risk": {"overall_risk_score": min(100, risk), "risk_level": rl, "flags": flags,
                "summary": f"{sum(1 for f in flags if f['detected'])} risk factors. Overall: {rl}."},
            "progress": None,
            "scout_summary": f"Scored {op}th percentile for {drill_type} among {age_group} athletes. Rating: {rating_from_pct(op)}. Injury risk: {rl}.",
        }
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

# ---------------------------------------------------------------------------
# Jersey color + team classification
# ---------------------------------------------------------------------------

def extract_jersey_color(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return np.array([0.0, 0.0, 0.0])
    mid_y = y1 + (y2 - y1) // 2
    crop  = frame[y1:mid_y, x1:x2]
    if crop.size == 0:
        return np.array([0.0, 0.0, 0.0])
    hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 10:
        return pixels.mean(axis=0)
    km = KMeans(n_clusters=1, n_init=3, random_state=42)
    km.fit(pixels)
    return km.cluster_centers_[0]

def classify_teams(tracker_ids, boxes, frame, color_memory):
    for tid, box in zip(tracker_ids, boxes):
        color = extract_jersey_color(frame, box)
        if tid not in color_memory:
            color_memory[tid] = color
        else:
            color_memory[tid] = 0.8 * color_memory[tid] + 0.2 * color
    if len(color_memory) < 3:
        return {tid: "home" for tid in tracker_ids}
    ids    = list(color_memory.keys())
    colors = np.array(list(color_memory.values()), dtype=np.float32)
    k      = min(3, len(ids))
    km     = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(colors)
    from collections import Counter
    counts  = Counter(labels)
    sorted_clusters = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
    cluster_to_team: dict[int, str] = {}
    if len(sorted_clusters) >= 3:
        cluster_to_team[sorted_clusters[0]] = "home"
        cluster_to_team[sorted_clusters[1]] = "away"
        cluster_to_team[sorted_clusters[2]] = "referee"
    elif len(sorted_clusters) == 2:
        cluster_to_team[sorted_clusters[0]] = "home"
        cluster_to_team[sorted_clusters[1]] = "away"
    else:
        cluster_to_team[sorted_clusters[0]] = "home"
    id_to_team = {tid: cluster_to_team.get(label, "home") for tid, label in zip(ids, labels)}
    return {tid: id_to_team.get(tid, "home") for tid in tracker_ids}

def detect_pitch_bounds(frame):
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([30, 40, 40]); upper = np.array([90, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)
    k     = np.ones((15, 15), np.uint8)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 1000:
        h, w = frame.shape[:2]; return 0, 0, w, h
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h

def pixel_to_pitch(px, py, pitch_bounds):
    x_min, y_min, x_max, y_max = pitch_bounds
    pw = max(x_max - x_min, 1); ph = max(y_max - y_min, 1)
    return max(0.0, min(1.0, (px - x_min) / pw)), max(0.0, min(1.0, (py - y_min) / ph))

def build_heatmap(positions):
    grid = [[0] * HEATMAP_COLS for _ in range(HEATMAP_ROWS)]
    for x_norm, y_norm in positions:
        col = min(int(x_norm * HEATMAP_COLS), HEATMAP_COLS - 1)
        row = min(int(y_norm * HEATMAP_ROWS), HEATMAP_ROWS - 1)
        grid[row][col] += 1
    return grid

def calculate_distance_m(positions):
    if len(positions) < 2: return 0.0
    total = 0.0
    for i in range(1, len(positions)):
        dx = (positions[i][0] - positions[i-1][0]) * PITCH_LENGTH_M
        dy = (positions[i][1] - positions[i-1][1]) * PITCH_WIDTH_M
        total += (dx**2 + dy**2) ** 0.5
    return round(total, 1)

def calculate_speeds(positions):
    speeds = []
    for i in range(1, len(positions)):
        dx = (positions[i][0] - positions[i-1][0]) * PITCH_LENGTH_M
        dy = (positions[i][1] - positions[i-1][1]) * PITCH_WIDTH_M
        speeds.append(round(((dx**2 + dy**2) ** 0.5) * 3.6, 1))
    return speeds

# ---------------------------------------------------------------------------
# POST /track
# ---------------------------------------------------------------------------

@app.post("/track")
async def track_video(
    file: UploadFile = File(...),
    squad: Optional[str] = Form(None),
) -> dict[str, Any]:
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    squad_map: dict[str, str] = {}
    if squad:
        try: squad_map = json.loads(squad)
        except json.JSONDecodeError: pass
    suffix = os.path.splitext(file.filename or "match.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read()); tmp_path = tmp.name
    try:
        return _run_tracking(tmp_path, squad_map)
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

def _run_tracking(video_path: str, squad_map: dict[str, str]) -> dict[str, Any]:
    model = get_model()
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Cannot open video file")
    original_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_every  = max(1, int(round(original_fps / SAMPLE_FPS)))
    tracker = sv.ByteTracker(
        track_activation_threshold=TRACKER_CONFIG.track_activation_threshold,
        lost_track_buffer=TRACKER_CONFIG.lost_track_buffer,
        minimum_matching_threshold=TRACKER_CONFIG.minimum_matching_threshold,
        frame_rate=SAMPLE_FPS,
        minimum_consecutive_frames=TRACKER_CONFIG.minimum_consecutive_frames,
    )
    player_positions: dict[int, list] = defaultdict(list)
    player_teams:     dict[int, str]  = {}
    player_seconds:   dict[int, list] = defaultdict(list)
    color_memory:     dict[int, Any]  = {}
    ball_positions:   list            = []
    last_ball_pos:    tuple | None    = None
    possession_frames = {"home": 0, "away": 0}
    pitch_bounds = None
    frame_idx = 0; second = 0; frames_processed = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % sample_every == 0:
            if frames_processed < 10:
                bounds = detect_pitch_bounds(frame)
                pitch_bounds = bounds if pitch_bounds is None else tuple(int(0.7*a+0.3*b) for a,b in zip(pitch_bounds, bounds))  # type: ignore
            if pitch_bounds is None:
                pitch_bounds = (0, 0, width, height)
            results      = model(frame, classes=[PERSON_CLASS_ID, BALL_CLASS_ID], verbose=False)[0]
            detections_all = sv.Detections.from_ultralytics(results)
            if detections_all.class_id is not None and len(detections_all) > 0:
                player_detections = detections_all[detections_all.class_id == PERSON_CLASS_ID]
                ball_detections   = detections_all[detections_all.class_id == BALL_CLASS_ID]
            else:
                player_detections = detections_all; ball_detections = sv.Detections.empty()
            player_detections = tracker.update_with_detections(player_detections)
            ball_pos_this_frame = None
            if len(ball_detections) > 0:
                best_idx = int(np.argmax(ball_detections.confidence)) if ball_detections.confidence is not None else 0
                bx1,by1,bx2,by2 = ball_detections.xyxy[best_idx]
                bx_norm, by_norm = pixel_to_pitch((bx1+bx2)/2, (by1+by2)/2, pitch_bounds)
                ball_pos_this_frame = (bx_norm, by_norm)
                last_ball_pos = ball_pos_this_frame
                ball_positions.append({"second": second, "x": round(bx_norm,3), "y": round(by_norm,3)})
            if len(player_detections) > 0 and player_detections.tracker_id is not None:
                tracker_ids = player_detections.tracker_id; boxes = player_detections.xyxy
                team_map    = classify_teams(tracker_ids, boxes, frame, color_memory)
                for tid, box in zip(tracker_ids, boxes):
                    px = (box[0]+box[2])/2.0; py = box[3]
                    x_norm, y_norm = pixel_to_pitch(px, py, pitch_bounds)
                    player_positions[int(tid)].append((x_norm, y_norm))
                    player_seconds[int(tid)].append(second)
                    player_teams[int(tid)] = team_map.get(int(tid), "home")
                ball_ref = ball_pos_this_frame or last_ball_pos
                if ball_ref:
                    min_dist = float("inf"); closest_team = "home"
                    for tid, box in zip(tracker_ids, boxes):
                        px=(box[0]+box[2])/2.0; py=box[3]
                        x_norm,y_norm = pixel_to_pitch(px,py,pitch_bounds)
                        dist = ((x_norm-ball_ref[0])**2+(y_norm-ball_ref[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist; closest_team = team_map.get(int(tid),"home")
                    if closest_team in ("home","away"):
                        possession_frames[closest_team] += 1
                else:
                    hc = sum(1 for tid in tracker_ids if team_map.get(int(tid))=="home")
                    ac = sum(1 for tid in tracker_ids if team_map.get(int(tid))=="away")
                    possession_frames["home" if hc >= ac else "away"] += 1
            second += 1; frames_processed += 1
        frame_idx += 1
    cap.release()
    players_out = []
    for tid, positions in player_positions.items():
        if len(positions) < 3: continue
        speeds = calculate_speeds(positions)
        players_out.append({
            "id": tid, "name": squad_map.get(str(tid), ""), "team": player_teams.get(tid, "home"),
            "positions": [{"second": s, "x": round(x,3), "y": round(y,3)} for s,(x,y) in zip(player_seconds[tid], positions)],
            "distance_m": calculate_distance_m(positions),
            "avg_x": round(sum(p[0] for p in positions)/len(positions), 3),
            "avg_y": round(sum(p[1] for p in positions)/len(positions), 3),
            "heatmap": build_heatmap(positions),
            "top_speed_kmh": round(max(speeds), 1) if speeds else 0.0,
            "avg_speed_kmh": round(sum(speeds)/len(speeds), 1) if speeds else 0.0,
        })
    total_poss = possession_frames["home"] + possession_frames["away"]
    poss_home  = round(possession_frames["home"] / total_poss * 100) if total_poss > 0 else 50
    return {
        "players": players_out, "ball": ball_positions,
        "stats": {"possession_home": poss_home, "possession_away": 100-poss_home,
            "duration_seconds": second, "frames_processed": frames_processed, "ball_detected_frames": len(ball_positions)},
        "video": {"width": width, "height": height, "fps": round(original_fps,2), "total_frames": total_frames},
    }

# ---------------------------------------------------------------------------
# Sprint detection + FFmpeg clips
# ---------------------------------------------------------------------------

def detect_sprint_events(players, threshold_kmh=SPRINT_THRESHOLD_KMH):
    events = []
    for player in players:
        positions = player.get("positions", [])
        for i in range(1, len(positions)):
            prev = positions[i-1]; curr = positions[i]
            dx = (curr["x"]-prev["x"]) * PITCH_LENGTH_M
            dy = (curr["y"]-prev["y"]) * PITCH_WIDTH_M
            speed_kmh = ((dx**2+dy**2)**0.5) * 3.6
            if speed_kmh >= threshold_kmh:
                events.append({"player_id": player["id"], "name": player.get("name",""), "team": player.get("team","home"), "second": curr["second"], "speed_kmh": round(speed_kmh,1)})
    events.sort(key=lambda e: e["speed_kmh"], reverse=True)
    return events

def clip_highlights(video_path, events, padding_s=5, max_clips=10):
    clips = []; seen_windows = []
    for event in events:
        if len(clips) >= max_clips: break
        second = event["second"]; start = max(0, second-padding_s); end = second+padding_s
        if any(not (end<=s or start>=e) for s,e in seen_windows): continue
        seen_windows.append((start, end))
        out_path = tempfile.mktemp(suffix=".mp4")
        cmd = ["ffmpeg","-y","-ss",str(start),"-i",video_path,"-t",str(end-start),"-c:v","libx264","-preset","fast","-crf","28","-c:a","aac","-movflags","+faststart",out_path]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0:
            clips.append({"event": event, "clip_path": out_path, "start": start, "end": end})
    return clips

def upload_clips_to_r2(clips):
    r2_account = os.environ.get("R2_ACCOUNT_ID"); r2_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret  = os.environ.get("R2_SECRET_ACCESS_KEY"); r2_bucket = os.environ.get("R2_BUCKET","grassroots-videos")
    r2_public  = os.environ.get("R2_PUBLIC_URL","").rstrip("/")
    s3_client  = None
    if all([r2_account, r2_key, r2_secret]):
        s3_client = boto3.client("s3", endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
            aws_access_key_id=r2_key, aws_secret_access_key=r2_secret, region_name="auto")
    results = []
    for clip in clips:
        key = f"highlights/{int(time.time())}-p{clip['event']['player_id']}-{clip['start']}s.mp4"
        url = ""
        try:
            if s3_client:
                s3_client.upload_file(clip["clip_path"], r2_bucket, key, ExtraArgs={"ContentType":"video/mp4"})
                url = f"{r2_public}/{key}" if r2_public else ""
        except Exception: url = ""
        finally:
            try: os.unlink(clip["clip_path"])
            except OSError: pass
        results.append({"event": clip["event"], "url": url, "key": key})
    return results

@app.post("/clip")
async def clip_video(
    file: UploadFile = File(...),
    squad: Optional[str] = Form(None),
    threshold_kmh: float = Form(25.0),
    max_clips: int = Form(10),
) -> dict[str, Any]:
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    squad_map: dict[str, str] = {}
    if squad:
        try: squad_map = json.loads(squad)
        except json.JSONDecodeError: pass
    suffix = os.path.splitext(file.filename or "match.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read()); tmp_path = tmp.name
    try:
        tracking = _run_tracking(tmp_path, squad_map)
        events   = detect_sprint_events(tracking["players"], threshold_kmh)
        clips    = clip_highlights(tmp_path, events, max_clips=max_clips)
        results  = upload_clips_to_r2(clips)
        return {"clips": results, "events_detected": len(events), "clips_generated": len(results), "stats": tracking["stats"]}
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------

class ProcessVideoRequest(BaseModel):
    video_key: str; match_id: int; callback_url: str

def download_from_r2(video_key: str) -> str:
    r2_account = os.environ.get("R2_ACCOUNT_ID"); r2_access = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret  = os.environ.get("R2_SECRET_ACCESS_KEY"); r2_bucket = os.environ.get("R2_BUCKET","grassroots-media")
    s3 = boto3.client("s3", endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
        aws_access_key_id=r2_access, aws_secret_access_key=r2_secret, region_name="auto")
    suffix = os.path.splitext(video_key)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False); s3.download_fileobj(r2_bucket, video_key, tmp); tmp.close()
    return tmp.name

def run_pipeline(video_key, match_id, callback_url):
    video_path = None
    try:
        video_path = download_from_r2(video_key)
        tracking   = _run_tracking(video_path, {})
        events     = detect_sprint_events(tracking["players"])
        raw_clips  = clip_highlights(video_path, events, max_clips=10)
        r2_account = os.environ.get("R2_ACCOUNT_ID"); r2_access = os.environ.get("R2_ACCESS_KEY_ID")
        r2_secret  = os.environ.get("R2_SECRET_ACCESS_KEY"); r2_bucket = os.environ.get("R2_BUCKET","grassroots-media")
        r2_public  = os.environ.get("R2_PUBLIC_URL","").rstrip("/")
        s3_client  = None
        if all([r2_account, r2_access, r2_secret]):
            s3_client = boto3.client("s3", endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
                aws_access_key_id=r2_access, aws_secret_access_key=r2_secret, region_name="auto")
        clips_out = []
        for clip in raw_clips:
            key = f"highlights/{match_id}/{int(time.time())}-p{clip['event']['player_id']}-{clip['start']}s.mp4"
            url = ""
            try:
                if s3_client:
                    s3_client.upload_file(clip["clip_path"], r2_bucket, key, ExtraArgs={"ContentType":"video/mp4"})
                    url = f"{r2_public}/{key}" if r2_public else ""
            except Exception: pass
            finally:
                try: os.unlink(clip["clip_path"])
                except OSError: pass
            clips_out.append({"player_id": clip["event"]["player_id"], "event_type": "sprint", "timestamp": clip["event"]["second"], "speed": clip["event"]["speed_kmh"], "url": url, "r2_key": key})
        with httpx.Client(timeout=30) as client:
            client.post(callback_url, json={"match_id": match_id, "status": "complete", "clips": clips_out})
    except Exception as exc:
        try:
            with httpx.Client(timeout=10) as client:
                client.post(callback_url, json={"match_id": match_id, "status": "failed", "error": str(exc), "clips": []})
        except Exception: pass
    finally:
        if video_path:
            try: os.unlink(video_path)
            except OSError: pass

@app.post("/process-video")
async def process_video(req: ProcessVideoRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    background_tasks.add_task(run_pipeline, req.video_key, req.match_id, req.callback_url)
    return {"status": "processing", "match_id": str(req.match_id)}

# ---------------------------------------------------------------------------
# Gemini File API proxy
# ---------------------------------------------------------------------------

@app.post("/gemini-upload")
async def gemini_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    google_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not google_key:
        raise HTTPException(status_code=500, detail="GOOGLE_AI_API_KEY not configured on AI service")
    content = await file.read()
    mime_type = file.content_type or "video/mp4"
    content_length = len(content)
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        init_res = await client.post(
            f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={google_key}",
            headers={"X-Goog-Upload-Protocol":"resumable","X-Goog-Upload-Command":"start",
                "X-Goog-Upload-Header-Content-Length":str(content_length),"X-Goog-Upload-Header-Content-Type":mime_type,"Content-Type":"application/json"},
            json={"file":{"display_name":f"match-{int(time.time())}"}})
        if init_res.status_code not in (200,201):
            raise HTTPException(status_code=502, detail=f"Gemini session init failed: {init_res.status_code}")
        upload_url = init_res.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise HTTPException(status_code=502, detail="Gemini did not return upload URL")
        upload_res = await client.put(upload_url,
            headers={"Content-Length":str(content_length),"X-Goog-Upload-Offset":"0","X-Goog-Upload-Command":"upload, finalize"},
            content=content)
        if upload_res.status_code not in (200,201):
            raise HTTPException(status_code=502, detail=f"Gemini upload failed: {upload_res.status_code}")
        file_info = upload_res.json().get("file",{})
        return {"fileUri": file_info.get("uri",""), "fileName": file_info.get("name",""), "mimeType": file_info.get("mimeType",mime_type), "state": file_info.get("state","ACTIVE")}

# ---------------------------------------------------------------------------
# Gemini match analysis (background)
# ---------------------------------------------------------------------------

class AnalyseRequest(BaseModel):
    fileUri: str; fileName: str; mimeType: str; fileState: Optional[str] = None
    homeTeam: str; awayTeam: str; competition: Optional[str] = ""; sport: Optional[str] = "football"

def _extract_json(text):
    try: return json.loads(text)
    except Exception: pass
    md = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if md:
        try: return json.loads(md.group(1))
        except Exception: pass
    obj = re.search(r"\{[\s\S]*\}", text)
    if obj:
        try: return json.loads(obj.group(0))
        except Exception: pass
    return None

async def _wait_for_file_active(file_name, google_key, job_id):
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0,read=60.0,write=10.0,pool=10.0)) as client:
        for _ in range(120):
            res = await client.get(f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={google_key}")
            if not res.is_success:
                raise RuntimeError(f"File state check failed: {res.status_code}")
            data = res.json(); state = data.get("state","")
            if state == "ACTIVE": return
            if state == "FAILED": raise RuntimeError("Gemini file processing failed")
            _jobs[job_id]["message"] = f"Gemini processing video... (state: {state})"
            await asyncio.sleep(5)
    raise RuntimeError("Video did not become ready within 10 minutes")

async def _analyse_background(job_id, req):
    try:
        google_key = os.environ.get("GOOGLE_AI_API_KEY","")
        if not google_key: raise RuntimeError("GOOGLE_AI_API_KEY not configured on AI service")
        _jobs[job_id]["message"] = "Waiting for Gemini to finish processing the video..."
        _jobs[job_id]["progress"] = 10
        if req.fileState != "ACTIVE":
            await _wait_for_file_active(req.fileName, google_key, job_id)
        _jobs[job_id]["message"] = "Gemini 2.0 Flash is watching the full match..."
        _jobs[job_id]["progress"] = 30
        system_prompt = (
            f"You are a professional football analyst with UEFA A-licence coaching experience.\n"
            f"Watch the full match: {req.homeTeam} vs {req.awayTeam}"
            + (f" ({req.competition})" if req.competition else "")
            + "\n\nReturn ONLY valid JSON with this structure:\n"
            '{"formation_home":"4-3-3","formation_away":"4-4-2","possession_home":55,"possession_away":45,'
            '"shots_home":8,"shots_away":5,"shots_on_target_home":4,"shots_on_target_away":2,"fouls_detected":3,'
            '"key_events":[{"time":"23:00","team":"home","type":"shot","description":"Right-footed shot"}],'
            '"tactical_patterns":[],"defensive_issues":[],"attacking_strengths":[],'
            '"man_of_match_candidate":"","halftime_recommendation":"","key_coaching_points":[]}'
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0,read=600.0,write=30.0,pool=10.0)) as client:
            gemini_res = await client.post(
                f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={google_key}",
                headers={"Content-Type":"application/json"},
                json={"contents":[{"parts":[{"text":system_prompt},{"file_data":{"mime_type":req.mimeType,"file_uri":req.fileUri}},{"text":"Provide your complete JSON analysis."}]}],
                    "generationConfig":{"temperature":0.2,"maxOutputTokens":4096}})
        _jobs[job_id]["progress"] = 75
        if not gemini_res.is_success:
            raise RuntimeError(f"Gemini API error: {gemini_res.status_code}")
        gemini_text = gemini_res.json().get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","")
        analysis = _extract_json(gemini_text)
        if analysis is None:
            raise RuntimeError(f"Gemini returned unreadable analysis: {gemini_text[:300]}")
        narrative = ""
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY","")
        if anthropic_key:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0,read=120.0,write=10.0,pool=10.0)) as client:
                claude_res = await client.post("https://api.anthropic.com/v1/messages",
                    headers={"x-api-key":anthropic_key,"anthropic-version":"2023-06-01","content-type":"application/json"},
                    json={"model":"claude-sonnet-4-6","max_tokens":1500,"messages":[{"role":"user","content":
                        f"You are a professional football analyst. Match: {req.homeTeam} vs {req.awayTeam}\n\n"
                        f"AI Analysis:\n{json.dumps(analysis,indent=2)}\n\n"
                        "Write a professional 4-paragraph tactical match report covering: overview, tactics, individual highlights, and training recommendations."}]})
            if claude_res.is_success:
                narrative = claude_res.json().get("content",[{}])[0].get("text","")
        _jobs[job_id].update({"status":"complete","progress":100,"message":"Analysis complete.","analysis":analysis,"narrative":narrative})
    except Exception as exc:
        _jobs[job_id].update({"status":"failed","error":str(exc),"progress":0})

@app.post("/analyse")
async def analyse_match(req: AnalyseRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    job_id = str(uuid_mod.uuid4())
    _jobs[job_id] = {"status":"processing","progress":0,"message":"Analysis queued...","analysis":None,"narrative":None,"error":None,"created_at":time.time()}
    background_tasks.add_task(_analyse_background, job_id, req)
    return {"job_id": job_id}

@app.get("/job/{job_id}")
async def get_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found — it may have expired")
    now = time.time()
    for k in [k for k,v in list(_jobs.items()) if now-v.get("created_at",now) > 7200]:
        _jobs.pop(k, None)
    return job

# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

class GenerateThumbnailRequest(BaseModel):
    video_url: str

@app.post("/generate-thumbnail")
async def generate_thumbnail(req: GenerateThumbnailRequest) -> dict[str, str]:
    tmp_video = None; tmp_thumb = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_video = f.name
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0,read=120.0,write=10.0,pool=5.0)) as client:
            async with client.stream("GET", req.video_url) as resp:
                if not resp.is_success:
                    raise HTTPException(status_code=502, detail=f"Could not download video: {resp.status_code}")
                with open(tmp_video, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024*256):
                        f.write(chunk)
        cap = cv2.VideoCapture(tmp_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps*3))
        ret, frame = cap.read(); cap.release()
        if not ret or frame is None:
            cap = cv2.VideoCapture(tmp_video); ret, frame = cap.read(); cap.release()
        if not ret or frame is None:
            raise HTTPException(status_code=422, detail="Could not extract frame from video")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_thumb = f.name
        cv2.imwrite(tmp_thumb, frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        r2_bucket = os.environ.get("R2_BUCKET","grassroots-videos")
        r2_public = os.environ.get("R2_PUBLIC_URL","").rstrip("/")
        thumb_key = f"thumbnails/fan-hub/{uuid_mod.uuid4()}.jpg"
        try:
            s3_client = boto3.client("s3",
                endpoint_url=f"https://{os.environ.get('R2_ACCOUNT_ID','')}.r2.cloudflarestorage.com",
                aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID",""),
                aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY",""), region_name="auto")
            s3_client.upload_file(tmp_thumb, r2_bucket, thumb_key, ExtraArgs={"ContentType":"image/jpeg"})
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"R2 upload failed: {e}")
        return {"thumbnail_url": f"{r2_public}/{thumb_key}" if r2_public else "", "r2_key": thumb_key}
    finally:
        for path in [tmp_video, tmp_thumb]:
            if path:
                try: os.unlink(path)
                except OSError: pass

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "grassroots-ai-tracker", "version": "3.0.0"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
