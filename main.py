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
# =============================================================================
# COACH ANALYSIS TOOLKIT — 6 tools for training ground use
# All endpoints under /coach-analysis/
# =============================================================================

# ---------------------------------------------------------------------------
# Tool 1 — Fatigue Tracker (MediaPipe)
# POST /coach-analysis/fatigue
#
# Coach films player doing 10+ reps of any movement.
# MediaPipe compares mechanics in first 3 reps vs last 3 reps.
# Returns: fatigue index %, which mechanics break down, substitution recommendation.
# ---------------------------------------------------------------------------

@app.post("/coach-analysis/fatigue")
async def analyse_fatigue(
    file: UploadFile = File(...),
    player_name: str = "Player",
    age_group:   str = "u17",
) -> dict[str, Any]:
    """
    Fatigue Tracker — compare technique in first vs last reps.

    How to film: Player does 10-15 reps of sprints, high knees, or circuits.
    Camera sideways, full body visible throughout.

    Returns fatigue index and substitution recommendation.
    """
    if age_group not in BENCHMARKS:
        age_group = "u17"
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    suffix = os.path.splitext(file.filename or "fatigue.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)

    try:
        frame_data, agg = process_video_with_mediapipe(tmp_path, target_fps=10)

        bench = BENCHMARKS[age_group]

        # Calculate degradation for each key metric
        degradation_metrics = []
        breakdown_details   = []

        metric_labels = {
            "left_knee_flexion":  "Left knee flexion",
            "right_knee_flexion": "Right knee flexion",
            "trunk_lean_deg":     "Trunk lean",
            "left_elbow_flexion": "Left arm drive",
            "right_elbow_flexion":"Right arm drive",
            "hip_drop_mm":        "Hip stability",
        }

        for key, label in metric_labels.items():
            d = agg.get(key, {})
            first = d.get("first_half_mean")
            last  = d.get("last_half_mean")
            if first and last and first > 0:
                pct_change = abs(last - first) / first * 100
                degradation_metrics.append(pct_change)
                if pct_change > 10:
                    direction = "decreased" if last < first else "increased"
                    breakdown_details.append({
                        "metric":       label,
                        "first_value":  round(first, 1),
                        "last_value":   round(last, 1),
                        "degradation":  round(pct_change, 1),
                        "detail": f"{label} {direction} by {round(pct_change, 1)}% from first to last reps. "
                                  f"{'This indicates muscle fatigue and loss of neuromuscular control.' if pct_change > 15 else 'Minor degradation — within acceptable range.'}"
                    })

        avg_fatigue_index = round(sum(degradation_metrics) / len(degradation_metrics), 1) if degradation_metrics else 0

        # Fatigue rating
        if avg_fatigue_index < 5:
            fatigue_rating = "Excellent"
            fatigue_color  = "green"
            sub_recommendation = "Full match — no substitution needed based on fatigue profile."
            match_minutes = 90
        elif avg_fatigue_index < 10:
            fatigue_rating = "Good"
            fatigue_color  = "green"
            sub_recommendation = "Can play full match. Monitor in second half."
            match_minutes = 90
        elif avg_fatigue_index < 18:
            fatigue_rating = "Moderate"
            fatigue_color  = "amber"
            sub_recommendation = "Manage minutes. Substitute between 60-75 minutes to prevent injury."
            match_minutes = 68
        elif avg_fatigue_index < 28:
            fatigue_rating = "High"
            fatigue_color  = "orange"
            sub_recommendation = "High fatigue detected. Substitute at half-time or by 60 minutes. Injury risk elevated in final 30 minutes."
            match_minutes = 55
        else:
            fatigue_rating = "Critical"
            fatigue_color  = "red"
            sub_recommendation = "Do not start. Player is showing significant fatigue. Rest and reassess in 48 hours."
            match_minutes = 0

        # Injury risk from fatigue
        injury_risk = avg_fatigue_index > 20
        injury_detail = (
            f"Technique degradation of {avg_fatigue_index}% significantly increases non-contact injury risk. "
            "Most non-contact injuries in football occur in the final 15-20 minutes of matches when players are fatigued."
        ) if injury_risk else None

        return {
            "tool":              "fatigue_tracker",
            "player_name":       player_name,
            "age_group":         age_group,
            "fatigue_index":     avg_fatigue_index,
            "fatigue_rating":    fatigue_rating,
            "fatigue_color":     fatigue_color,
            "frames_analysed":   len(frame_data),
            "breakdown_details": breakdown_details,
            "substitution_recommendation": sub_recommendation,
            "recommended_match_minutes":   match_minutes,
            "injury_risk":       injury_risk,
            "injury_detail":     injury_detail,
            "what_was_measured": (
                f"MediaPipe BlazePose tracked {len(frame_data)} frames. "
                f"Compared body mechanics in the first half of the clip vs the second half. "
                f"Measured: knee flexion, trunk lean, arm drive, hip stability. "
                f"Average technique degradation: {avg_fatigue_index}%."
            ),
            "coach_action": (
                f"Based on this fatigue profile, {player_name} should "
                + ("start and play the full match." if match_minutes >= 90
                   else f"be substituted around {match_minutes} minutes."
                   if match_minutes > 0
                   else "rest and not play today.")
            ),
        }

    finally:
        try: os.unlink(tmp_path)
        except OSError: pass


# ---------------------------------------------------------------------------
# Tool 2 — Injury Risk Screener (MediaPipe)
# POST /coach-analysis/injury-risk
#
# Coach films player doing single-leg landing test.
# MediaPipe detects knee valgus, hip drop, landing stiffness.
# Returns: traffic light risk level per leg + specific flags.
# ---------------------------------------------------------------------------

@app.post("/coach-analysis/injury-risk")
async def analyse_injury_risk(
    file: UploadFile = File(...),
    player_name: str = "Player",
    age_group:   str = "u17",
) -> dict[str, Any]:
    """
    Injury Risk Screener — single-leg landing test.

    How to film: Player jumps off two feet and lands on one leg, 5 times each leg.
    Camera directly in front, full body visible from head to toe.

    Returns: ACL risk, hip stability, bilateral asymmetry — colour-coded.
    """
    if age_group not in BENCHMARKS:
        age_group = "u17"
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    suffix = os.path.splitext(file.filename or "injury.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)

    try:
        frame_data, agg = process_video_with_mediapipe(tmp_path, target_fps=10)
        bench = BENCHMARKS[age_group]

        risk_score = 0
        flags      = []

        # ── Knee valgus (ACL risk) ─────────────────────────────────────────
        l_val = abs(agg.get("left_knee_valgus_mm",  {}).get("mean", 0))
        r_val = abs(agg.get("right_knee_valgus_mm", {}).get("mean", 0))
        val_bench = bench["balance_knee_valgus"]

        for side, val in [("Left", l_val), ("Right", r_val)]:
            if val > val_bench["needs_work"]:
                sev = "Critical"; pts = 40
            elif val > val_bench["good"]:
                sev = "High"; pts = 25
            elif val > val_bench["elite"]:
                sev = "Moderate"; pts = 12
            else:
                sev = "Low"; pts = 0
            risk_score += pts
            flags.append({
                "flag":        f"{side} Knee Valgus (ACL Risk)",
                "severity":    sev,
                "detected":    val > val_bench["elite"],
                "measurement": f"{round(val)}mm inward collapse",
                "detail": (
                    f"The {side.lower()} knee collapsed {round(val)}mm inward during landing. "
                    f"Safe threshold for {age_group}: under {val_bench['elite']}mm. "
                    f"{'CRITICAL: This level of knee valgus is the primary biomechanical predictor of ACL tears. Do not play competitive matches until resolved.' if sev == 'Critical' else 'Knee valgus detected. Strengthening hip abductors will reduce this significantly within 6-8 weeks.' if sev == 'High' else 'Minor valgus noted. Monitor and include hip strengthening in warm-up routine.' if sev == 'Moderate' else 'Good knee alignment — safe to play.'}"
                ),
                "fix": [
                    "Lateral band walks — 3 sets of 15 reps daily",
                    "Clamshells — 3 sets of 20 reps",
                    "Single-leg Romanian deadlift — 3 sets of 10 per leg",
                    "Wall squat with resistance band around knees",
                ] if sev in ("Critical", "High") else [],
                "timeline": "6-8 weeks of consistent hip strengthening" if sev in ("Critical", "High") else None,
            })

        # ── Hip drop (IT band / hip injury) ───────────────────────────────
        hip_drop  = agg.get("hip_drop_mm", {}).get("mean", 0)
        drop_bench = bench["balance_hip_drop"]

        if hip_drop > drop_bench["needs_work"]:
            sev = "High"; pts = 20
        elif hip_drop > drop_bench["good"]:
            sev = "Moderate"; pts = 10
        elif hip_drop > drop_bench["elite"]:
            sev = "Low"; pts = 4
        else:
            sev = "None"; pts = 0
        risk_score += pts

        flags.append({
            "flag":        "Hip Drop (Trendelenburg Sign)",
            "severity":    sev,
            "detected":    hip_drop > drop_bench["elite"],
            "measurement": f"{round(hip_drop)}mm hip height difference",
            "detail": (
                f"Hip dropped {round(hip_drop)}mm on one side during single-leg landing. "
                f"{'Significant hip drop detected — indicates weak gluteus medius. Strongly associated with IT band syndrome, patellofemoral pain, and hip flexor injury.' if sev == 'High' else 'Moderate hip drop detected — monitor and include glute activation in warm-up.' if sev == 'Moderate' else 'Minor hip drop — within acceptable range.' if sev == 'Low' else 'Excellent hip stability. Glutes are doing their job correctly.'}"
            ),
            "fix": [
                "Single-leg stance balance — 3 sets of 30 seconds per leg",
                "Glute medius side-lying leg raises — 3 sets of 20",
                "Hip hike exercise on a step — 3 sets of 15 per side",
                "Single-leg squat with hip focus",
            ] if sev in ("High", "Moderate") else [],
        })

        # ── Landing stiffness (joint stress) ──────────────────────────────
        l_knee_min = agg.get("left_knee_flexion",  {}).get("min", 150)
        r_knee_min = agg.get("right_knee_flexion", {}).get("min", 150)
        avg_min_knee = (l_knee_min + r_knee_min) / 2

        stiff_bench = bench["jump_landing_knee"]
        if avg_min_knee > stiff_bench["needs_work"]:
            sev = "High"; pts = 18
        elif avg_min_knee > stiff_bench["good"]:
            sev = "Moderate"; pts = 8
        elif avg_min_knee > stiff_bench["elite"]:
            sev = "Low"; pts = 3
        else:
            sev = "None"; pts = 0
        risk_score += pts

        flags.append({
            "flag":        "Landing Stiffness",
            "severity":    sev,
            "detected":    avg_min_knee > stiff_bench["good"],
            "measurement": f"Minimum knee bend: {round(avg_min_knee)}°",
            "detail": (
                f"Minimum knee flexion on landing: {round(avg_min_knee)}°. "
                f"Target for {age_group}: under {stiff_bench['elite']}°. "
                f"{'Stiff landing detected — knees barely bending on impact. This transfers excessive force to knee and ankle joints, increasing stress fracture and cartilage damage risk.' if sev == 'High' else 'Moderate stiffness — more knee bend on landing will reduce injury risk.' if sev == 'Moderate' else 'Good landing mechanics — adequate knee absorption.' if sev == 'None' else 'Minor stiffness noted.'}"
            ),
            "fix": [
                "Box landing drills — focus on soft, bent-knee landings",
                "Drop squat practice — land in a deep squat position",
                "Depth jumps with coaching cue 'quiet landing'",
            ] if sev in ("High", "Moderate") else [],
        })

        # ── Bilateral asymmetry ────────────────────────────────────────────
        l_knee_mean = agg.get("left_knee_flexion",  {}).get("mean", 120)
        r_knee_mean = agg.get("right_knee_flexion", {}).get("mean", 120)
        if max(l_knee_mean, r_knee_mean) > 0:
            asym_pct = abs(l_knee_mean - r_knee_mean) / max(l_knee_mean, r_knee_mean) * 100
            asym_bench = bench["sprint_asymmetry"]
            dominant   = "Left" if l_knee_mean > r_knee_mean else "Right"
            weaker     = "Right" if dominant == "Left" else "Left"

            if asym_pct > asym_bench["needs_work"]:
                sev = "High"; pts = 15
            elif asym_pct > asym_bench["good"]:
                sev = "Moderate"; pts = 7
            elif asym_pct > asym_bench["elite"]:
                sev = "Low"; pts = 2
            else:
                sev = "None"; pts = 0
            risk_score += pts

            flags.append({
                "flag":        "Bilateral Asymmetry",
                "severity":    sev,
                "detected":    asym_pct > asym_bench["elite"],
                "measurement": f"{round(asym_pct, 1)}% difference between legs",
                "detail": (
                    f"{dominant} leg is {round(asym_pct, 1)}% stronger than {weaker} leg. "
                    f"{'Significant asymmetry — the dominant leg is overcompensating, increasing overuse injury risk on that side over a full season.' if sev == 'High' else 'Moderate asymmetry — include single-leg work targeting the weaker side.' if sev == 'Moderate' else 'Good bilateral symmetry.' if sev == 'None' else 'Minor asymmetry — normal range.'}"
                ),
                "fix": [
                    f"Single-leg exercises focusing exclusively on {weaker} leg for 4 weeks",
                    "Bulgarian split squat — 4 sets on weaker side, 2 on stronger",
                    "Single-leg hop and hold on weaker side",
                ] if sev in ("High", "Moderate") else [],
            })

        # ── Overall risk level ─────────────────────────────────────────────
        if risk_score >= 70:
            risk_level = "Critical"; risk_color = "red"
            play_decision = "DO NOT PLAY. High ACL and overuse injury risk detected. Refer to physio before next training session."
        elif risk_score >= 45:
            risk_level = "High"; risk_color = "orange"
            play_decision = "Restricted training only. No competitive matches until flags are resolved. Start strengthening programme immediately."
        elif risk_score >= 25:
            risk_level = "Moderate"; risk_color = "amber"
            play_decision = "Can train and play but monitor closely. Include targeted strengthening in daily warm-up. Reassess in 3 weeks."
        elif risk_score >= 10:
            risk_level = "Low"; risk_color = "yellow"
            play_decision = "Low risk. Include general hip and glute strengthening as prevention. Clear to play."
        else:
            risk_level = "Excellent"; risk_color = "green"
            play_decision = "No injury risk flags detected. Player is biomechanically safe to train and compete fully."

        return {
            "tool":         "injury_risk_screener",
            "player_name":  player_name,
            "age_group":    age_group,
            "risk_score":   min(100, risk_score),
            "risk_level":   risk_level,
            "risk_color":   risk_color,
            "play_decision": play_decision,
            "flags":        flags,
            "frames_analysed": len(frame_data),
            "what_was_measured": (
                "MediaPipe BlazePose tracked 33 body landmarks across all frames. "
                "Measured: knee valgus (ACL predictor), hip drop (IT band predictor), "
                "landing stiffness (joint stress), bilateral leg symmetry. "
                "Compared to age-group benchmarks from sports medicine research."
            ),
            "coach_action": play_decision,
            "priority_fix": next(
                (f["flag"] for f in flags if f["severity"] in ("Critical", "High") and f["detected"]),
                "No urgent issues — maintain general conditioning."
            ),
        }

    finally:
        try: os.unlink(tmp_path)
        except OSError: pass


# ---------------------------------------------------------------------------
# Tool 3 — First Touch Analyser (Gemini)
# POST /coach-analysis/first-touch
#
# Coach films player receiving 10 passes from different angles.
# Gemini sees ball + player, scores each touch.
# Returns: touch quality per rep, average, dominant foot, body shape rating.
# ---------------------------------------------------------------------------

@app.post("/coach-analysis/first-touch")
async def analyse_first_touch(
    file: UploadFile = File(...),
    player_name: str = "Player",
    age_group:   str = "u17",
) -> dict[str, Any]:
    """
    First Touch Analyser — Gemini watches player receiving passes.

    How to film: Player receives 10 passes from different angles and distances.
    Film from the side so body shape and ball trajectory are both visible.

    Returns: touch quality scores, dominant foot, body shape analysis.
    """
    google_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not google_key:
        raise HTTPException(status_code=500, detail="GOOGLE_AI_API_KEY not configured")

    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Upload to Gemini File API
    content   = await file.read()
    mime_type = file.content_type or "video/mp4"

    timeout = httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        init_res = await client.post(
            f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={google_key}",
            headers={
                "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(content)),
                "X-Goog-Upload-Header-Content-Type": mime_type, "Content-Type": "application/json",
            },
            json={"file": {"display_name": f"first-touch-{int(time.time())}"}},
        )
        if not init_res.is_success:
            raise HTTPException(status_code=502, detail="Gemini upload failed")

        upload_url = init_res.headers.get("X-Goog-Upload-URL")
        upload_res = await client.put(upload_url,
            headers={"Content-Length": str(len(content)), "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"},
            content=content)

        file_uri  = upload_res.json().get("file", {}).get("uri", "")

        # Wait for Gemini to process
        for _ in range(30):
            check = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/files/{file_uri.split('/')[-1]}?key={google_key}"
            )
            if check.json().get("state") == "ACTIVE":
                break
            await asyncio.sleep(3)

        # Analyse with Gemini
        prompt = f"""You are a professional football coach analysing a player's first touch.
Watch this video of {player_name} (age group: {age_group}) receiving passes.

Analyse every touch in the video carefully. Return ONLY valid JSON:
{{
  "total_touches_observed": 10,
  "dominant_foot": "Right",
  "overall_score": 7.2,
  "touches": [
    {{
      "touch_number": 1,
      "score": 8,
      "foot_used": "Right",
      "surface": "Inside of foot",
      "ball_control": "Good — ball stayed within 1 metre",
      "body_shape": "Open — able to play forward immediately",
      "weakness": null
    }}
  ],
  "strengths": ["Consistent inside-foot control", "Good body shape on right side"],
  "weaknesses": ["Left foot receiving needs work — body closes off play", "First touch too heavy on aerial balls"],
  "technical_rating": "Developing",
  "specific_fix": "When receiving on left side, open hips earlier before ball arrives. This creates space and allows forward play.",
  "coach_instruction": "Drill recommendation: receiving and turning drill with cones, 15 minutes daily focusing on left side."
}}

Score each touch 1-10:
10 = Perfect control, ideal body shape, immediate forward option
7-9 = Good touch, minor adjustment needed
4-6 = Touch retained but suboptimal position or direction
1-3 = Heavy touch, ball lost or major recovery needed"""

        gemini_res = await client.post(
            f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={google_key}",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [
                {"text": prompt},
                {"file_data": {"mime_type": mime_type, "file_uri": file_uri}},
                {"text": "Provide your complete JSON analysis of all touches in this video."}
            ]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2000}},
        )

    if not gemini_res.is_success:
        raise HTTPException(status_code=502, detail="Gemini analysis failed")

    text     = gemini_res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    analysis = _extract_json(text)

    if not analysis:
        raise HTTPException(status_code=422, detail="Could not parse Gemini response")

    overall = analysis.get("overall_score", 5)
    if overall >= 8.5:   rating = "Elite"
    elif overall >= 7:   rating = "Advanced"
    elif overall >= 5.5: rating = "Developing"
    elif overall >= 4:   rating = "Foundation"
    else:                rating = "Beginner"

    return {
        "tool":          "first_touch_analyser",
        "player_name":   player_name,
        "age_group":     age_group,
        "overall_score": overall,
        "rating":        rating,
        "analysis":      analysis,
        "what_was_measured": (
            "Gemini 2.0 Flash watched the full video and scored each individual touch. "
            "Measured: foot surface used, ball control distance, body shape on receiving, "
            "forward play options created, dominant foot identification."
        ),
        "coach_action": analysis.get("coach_instruction", "Focus on weak foot receiving drills."),
    }


# ---------------------------------------------------------------------------
# Tool 4 — Sprint Mechanics Report (MediaPipe)
# POST /coach-analysis/sprint-mechanics
#
# Coach films player sprinting from the side (3 runs minimum).
# MediaPipe measures trunk lean, knee drive, arm drive, stride symmetry.
# Returns: mechanics breakdown + specific fix per weakness.
# ---------------------------------------------------------------------------

@app.post("/coach-analysis/sprint-mechanics")
async def analyse_sprint_mechanics(
    file: UploadFile = File(...),
    player_name: str = "Player",
    age_group:   str = "u17",
) -> dict[str, Any]:
    """
    Sprint Mechanics Report — MediaPipe analyses sprinting technique.

    How to film: Player sprints past the camera (sideways view) 3 times.
    Camera at hip height for best landmark detection.
    Film all 3 runs in one continuous clip.

    Returns: trunk lean, knee drive, arm drive, stride symmetry scores.
    """
    if age_group not in BENCHMARKS:
        age_group = "u17"
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    suffix = os.path.splitext(file.filename or "sprint.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)

    try:
        frame_data, agg = process_video_with_mediapipe(tmp_path, target_fps=10)
        bench = BENCHMARKS[age_group]

        metrics     = []
        total_score = []

        # ── Trunk lean ──────────────────────────────────────────────────────
        trunk_max = agg.get("trunk_lean_deg", {}).get("max", 10)
        lean_pct  = percentile_from_bench(trunk_max, bench["sprint_trunk_lean"], higher_is_better=True)
        total_score.append(lean_pct)
        metrics.append({
            "metric":    "Trunk Forward Lean",
            "value":     f"{round(trunk_max, 1)}°",
            "benchmark": f"{bench['sprint_trunk_lean']['elite']}° (elite) / {bench['sprint_trunk_lean']['good']}° (good)",
            "percentile": lean_pct,
            "rating":    rating_from_pct(lean_pct),
            "detail": (
                f"Maximum trunk lean during acceleration: {round(trunk_max, 1)}°. "
                f"{'Excellent forward lean — driving force efficiently into the ground.' if lean_pct >= 70 else 'Insufficient lean — upright posture wastes energy going upward instead of forward. A 5-10° more aggressive lean will significantly improve acceleration.'}"
            ),
            "fix": "Acceleration wall drills — lean at 45° against a wall, drive knees up for 10 reps. Train the body to feel the forward lean position." if lean_pct < 70 else None,
        })

        # ── Stride symmetry ──────────────────────────────────────────────────
        l_knee = agg.get("left_knee_flexion",  {}).get("mean")
        r_knee = agg.get("right_knee_flexion", {}).get("mean")
        if l_knee and r_knee and max(l_knee, r_knee) > 0:
            asym = abs(l_knee - r_knee) / max(l_knee, r_knee) * 100
            sym_pct = percentile_from_bench(asym, bench["sprint_asymmetry"], higher_is_better=False)
            dominant = "Left" if l_knee > r_knee else "Right"
            total_score.append(sym_pct)
            metrics.append({
                "metric":    "Stride Symmetry",
                "value":     f"{round(100 - asym, 1)}% symmetric ({round(asym, 1)}% L/R difference)",
                "benchmark": f"Under {bench['sprint_asymmetry']['elite']}% difference (elite)",
                "percentile": sym_pct,
                "rating":    rating_from_pct(sym_pct),
                "detail": (
                    f"Left-right stride asymmetry: {round(asym, 1)}%. "
                    f"{'Excellent symmetry — both legs contributing equally.' if sym_pct >= 70 else f'{dominant} leg is dominant. The weaker leg reduces stride efficiency and causes the dominant side to overwork, increasing overuse injury risk over a season.'}"
                ),
                "fix": f"Single-leg bounding on weaker leg — 3 sets of 8 bounds. Hip flexor stretching on weaker side daily." if sym_pct < 70 else None,
            })

        # ── Knee drive height ────────────────────────────────────────────────
        l_hip_max = agg.get("left_hip_flexion",  {}).get("max", 60)
        r_hip_max = agg.get("right_hip_flexion", {}).get("max", 60)
        avg_knee_drive = (l_hip_max + r_hip_max) / 2
        # Higher hip flexion = better knee drive
        drive_pct = percentile_from_bench(avg_knee_drive, {"elite": 80, "good": 65, "needs_work": 50}, higher_is_better=True)
        total_score.append(drive_pct)
        metrics.append({
            "metric":    "Knee Drive Height",
            "value":     f"{round(avg_knee_drive, 1)}° hip flexion",
            "benchmark": "80°+ (elite) / 65°+ (good)",
            "percentile": drive_pct,
            "rating":    rating_from_pct(drive_pct),
            "detail": (
                f"Average knee drive (hip flexion): {round(avg_knee_drive, 1)}°. "
                f"{'Excellent knee drive — generating good stride length and power.' if drive_pct >= 70 else 'Low knee drive detected. Higher knee drive increases stride length and running speed. Elite sprinters drive the knee to 80°+ hip flexion on every stride.'}"
            ),
            "fix": "High knee drills — 3 sets of 20 metres. A-skips and B-skips for knee drive mechanics." if drive_pct < 70 else None,
        })

        # ── Arm drive ────────────────────────────────────────────────────────
        l_elbow = agg.get("left_elbow_flexion",  {}).get("mean", 90)
        r_elbow = agg.get("right_elbow_flexion", {}).get("mean", 90)
        avg_elbow = (l_elbow + r_elbow) / 2
        # Optimal arm drive = 85-95° elbow flexion maintained throughout
        elbow_diff = abs(avg_elbow - 90)
        arm_pct = max(20, 95 - int(elbow_diff * 3))
        total_score.append(arm_pct)
        metrics.append({
            "metric":    "Arm Drive Mechanics",
            "value":     f"{round(avg_elbow, 1)}° average elbow angle",
            "benchmark": "85-95° elbow angle (elite — efficient arm drive)",
            "percentile": arm_pct,
            "rating":    rating_from_pct(arm_pct),
            "detail": (
                f"Average elbow flexion during sprint: {round(avg_elbow, 1)}°. "
                f"{'Good arm drive — elbows at efficient angle.' if arm_pct >= 70 else 'Arm mechanics need attention. Arms should be at ~90° elbow angle and drive forward-back (not across the body). Poor arm mechanics reduce running efficiency by up to 10%.'}"
            ),
            "fix": "Arm drive seated drill — sit on ground, practice arm pumping at 90°. Then standing, eyes closed, focusing on straight forward-back motion." if arm_pct < 70 else None,
        })

        overall = round(sum(total_score) / len(total_score)) if total_score else 50

        return {
            "tool":          "sprint_mechanics_report",
            "player_name":   player_name,
            "age_group":     age_group,
            "overall_score": overall,
            "overall_rating": rating_from_pct(overall),
            "metrics":       metrics,
            "frames_analysed": len(frame_data),
            "what_was_measured": (
                "MediaPipe BlazePose measured sprint mechanics across all frames. "
                "Trunk lean: how aggressively the player drives forward during acceleration. "
                "Stride symmetry: whether left and right legs contribute equally. "
                "Knee drive: how high the knee lifts on each stride (determines stride length). "
                "Arm drive: elbow angle and movement efficiency."
            ),
            "priority_fix": next(
                (m["fix"] for m in metrics if m.get("fix") and m["percentile"] < 55),
                "No urgent mechanical fixes needed."
            ),
            "coach_action": (
                f"{player_name}'s sprint mechanics scored {overall}th percentile. "
                + ("Excellent mechanics — focus on maintaining form under fatigue." if overall >= 70
                   else "Focus on: " + ", ".join(m["metric"] for m in metrics if m["percentile"] < 55) + " in the next training block.")
            ),
        }

    finally:
        try: os.unlink(tmp_path)
        except OSError: pass


# ---------------------------------------------------------------------------
# Tool 5 — Set Piece Technique (Gemini)
# POST /coach-analysis/set-piece
#
# Coach films player taking free kicks, corners, shots, or headers.
# Gemini sees ball + player, analyses contact quality, plant foot, hip rotation.
# ---------------------------------------------------------------------------

@app.post("/coach-analysis/set-piece")
async def analyse_set_piece(
    file:       UploadFile = File(...),
    player_name: str = "Player",
    set_piece_type: str = "shooting",  # shooting | freekick | corner | header | penalty
    age_group:   str = "u17",
) -> dict[str, Any]:
    """
    Set Piece Technique Analyser — Gemini watches ball contact and body mechanics.

    How to film: Player takes 5-10 repetitions of the same set piece.
    Film from the side or slightly behind for best view of plant foot and follow-through.

    Returns: plant foot analysis, hip rotation, contact quality, follow-through.
    """
    google_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not google_key:
        raise HTTPException(status_code=500, detail="GOOGLE_AI_API_KEY not configured")

    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    content   = await file.read()
    mime_type = file.content_type or "video/mp4"

    timeout = httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        init_res = await client.post(
            f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={google_key}",
            headers={
                "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(content)),
                "X-Goog-Upload-Header-Content-Type": mime_type, "Content-Type": "application/json",
            },
            json={"file": {"display_name": f"set-piece-{int(time.time())}"}},
        )
        upload_url = init_res.headers.get("X-Goog-Upload-URL")
        upload_res = await client.put(upload_url,
            headers={"Content-Length": str(len(content)), "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"},
            content=content)
        file_uri = upload_res.json().get("file", {}).get("uri", "")

        for _ in range(30):
            check = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/files/{file_uri.split('/')[-1]}?key={google_key}"
            )
            if check.json().get("state") == "ACTIVE":
                break
            await asyncio.sleep(3)

        type_prompts = {
            "shooting": "shooting technique — plant foot placement, hip rotation, contact point on ball, follow-through",
            "freekick": "free kick technique — run-up angle, plant foot, strike contact point, ball flight shape",
            "corner":   "corner kick delivery — body angle, swing of the leg, contact point, ball flight and spin",
            "header":   "heading technique — jump timing, neck position, contact point on forehead, direction",
            "penalty":  "penalty technique — approach run, plant foot, contact point, goalkeeper reading",
        }
        focus = type_prompts.get(set_piece_type, type_prompts["shooting"])

        prompt = f"""You are a UEFA A-licence coach analysing {set_piece_type} technique.
Watch {player_name} (age group: {age_group}) performing {set_piece_type} repetitions.
Focus on: {focus}

Return ONLY valid JSON:
{{
  "total_reps_observed": 6,
  "dominant_foot": "Right",
  "overall_score": 6.8,
  "reps": [
    {{
      "rep_number": 1,
      "score": 7,
      "plant_foot_distance_cm": "Correct — 20cm to the side of ball",
      "plant_foot_angle": "Good — pointing toward target",
      "hip_rotation": "Partial — could open more before contact",
      "contact_point": "Inside of foot — correct surface",
      "follow_through": "Short — cut off early, reducing power",
      "ball_flight": "Accurate but lacking pace",
      "weakness": "Follow-through terminated early"
    }}
  ],
  "consistent_strengths": ["Good plant foot placement", "Accurate contact point"],
  "consistent_weaknesses": ["Hip rotation incomplete", "Follow-through cut short"],
  "technical_rating": "Developing",
  "single_biggest_fix": "Allow the kicking leg to follow through fully toward the target after contact. Stopping the follow-through early reduces power by 20-30%.",
  "drill_recommendation": "Follow-through practice — kick through a target 50cm past the ball. Do 20 reps focusing only on completing the follow-through fully."
}}

Be specific and technical. Reference exact body positions you observe."""

        gemini_res = await client.post(
            f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={google_key}",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [
                {"text": prompt},
                {"file_data": {"mime_type": mime_type, "file_uri": file_uri}},
                {"text": "Provide your complete JSON technical analysis."}
            ]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2000}},
        )

    if not gemini_res.is_success:
        raise HTTPException(status_code=502, detail="Gemini analysis failed")

    text     = gemini_res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    analysis = _extract_json(text)

    if not analysis:
        raise HTTPException(status_code=422, detail="Could not parse Gemini response")

    overall = analysis.get("overall_score", 5)

    return {
        "tool":           "set_piece_technique",
        "player_name":    player_name,
        "set_piece_type": set_piece_type,
        "age_group":      age_group,
        "overall_score":  overall,
        "rating":         rating_from_pct(int(overall * 10)),
        "analysis":       analysis,
        "what_was_measured": (
            f"Gemini 2.0 Flash watched all {set_piece_type} repetitions. "
            "Measured: plant foot placement and angle relative to ball, "
            "hip rotation timing and degree, ball contact point (which part of foot/head), "
            "follow-through direction and length, resulting ball flight."
        ),
        "coach_action": analysis.get("drill_recommendation", "Focus on technique consistency."),
        "priority_fix": analysis.get("single_biggest_fix", "Work on consistency across repetitions."),
    }


# ---------------------------------------------------------------------------
# Tool 6 — Match Readiness Score (MediaPipe — 3 clips)
# POST /coach-analysis/match-readiness
#
# Coach films 3 quick clips: jump + sprint + balance.
# MediaPipe combines all three into one readiness percentage.
# Returns: overall % + colour recommendation + per-component scores.
# ---------------------------------------------------------------------------

@app.post("/coach-analysis/match-readiness")
async def analyse_match_readiness(
    jump_file:    UploadFile = File(...),
    sprint_file:  UploadFile = File(...),
    balance_file: UploadFile = File(...),
    player_name:  str = "Player",
    age_group:    str = "u17",
) -> dict[str, Any]:
    """
    Match Readiness Score — 3 quick clips combined into one readiness %.

    How to film:
    - Jump clip: 5 standing jumps, sideways view (30 seconds)
    - Sprint clip: 3 sprints past camera, sideways view (20 seconds)
    - Balance clip: single-leg balance 15s each leg, front view (40 seconds)

    Returns: overall readiness % with colour-coded play recommendation.
    """
    if age_group not in BENCHMARKS:
        age_group = "u17"

    results = {}
    tmp_paths = []

    async def process_clip(upload_file: UploadFile, test_name: str, scorer_fn):
        suffix = os.path.splitext(upload_file.filename or f"{test_name}.mp4")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp_paths.append(tmp_path)
            while chunk := await upload_file.read(1024 * 1024):
                tmp.write(chunk)
        try:
            _, agg = process_video_with_mediapipe(tmp_path, target_fps=10)
            return scorer_fn(agg, age_group)
        except Exception as e:
            return {"test_score": 50, "rating": "Unknown", "detail": f"Could not process {test_name} clip: {str(e)}"}

    try:
        jump_result    = await process_clip(jump_file,    "jump",    score_jump)
        sprint_result  = await process_clip(sprint_file,  "sprint",  score_sprint)
        balance_result = await process_clip(balance_file, "balance", score_balance)

        # Weighted readiness score
        # Balance weighted highest — most sensitive to injury/fatigue
        readiness = round(
            jump_result["test_score"]    * 0.30 +
            sprint_result["test_score"]  * 0.30 +
            balance_result["test_score"] * 0.40
        )

        # Colour-coded recommendation
        if readiness >= 85:
            color      = "green"
            label      = "Full Match Ready"
            decision   = f"{player_name} is fully ready to play. No restrictions. Full 90 minutes recommended."
            minutes    = 90
        elif readiness >= 70:
            color      = "green"
            label      = "Ready — Monitor"
            decision   = f"{player_name} is ready to play. Monitor during the match. Substitute if performance drops after 70 minutes."
            minutes    = 90
        elif readiness >= 55:
            color      = "amber"
            label      = "Manage Minutes"
            decision   = f"{player_name} should play but manage minutes. Substitute at 60-65 minutes to prevent injury risk in the final phase."
            minutes    = 63
        elif readiness >= 40:
            color      = "orange"
            label      = "Limited Role Only"
            decision   = f"{player_name} is not fully ready. Consider a cameo role from the bench (20-30 minutes) or leave out if there is a viable alternative."
            minutes    = 25
        else:
            color      = "red"
            label      = "Do Not Play"
            decision   = f"{player_name} is not ready to play. Biomechanical indicators suggest elevated injury risk. Rest and reassess in 48 hours."
            minutes    = 0

        # Flags from balance (most sensitive to readiness issues)
        readiness_flags = []
        if balance_result.get("injury_flag"):
            readiness_flags.append(balance_result.get("injury_detail", "Injury risk flag from balance test"))
        if jump_result.get("injury_flag"):
            readiness_flags.append(jump_result.get("injury_detail", "Landing mechanics concern"))

        return {
            "tool":              "match_readiness_score",
            "player_name":       player_name,
            "age_group":         age_group,
            "readiness_score":   readiness,
            "readiness_color":   color,
            "readiness_label":   label,
            "play_decision":     decision,
            "recommended_minutes": minutes,
            "components": {
                "jump": {
                    "score":  jump_result["test_score"],
                    "rating": jump_result["rating"],
                    "detail": jump_result["detail"],
                    "weight": "30%",
                },
                "sprint": {
                    "score":  sprint_result["test_score"],
                    "rating": sprint_result["rating"],
                    "detail": sprint_result["detail"],
                    "weight": "30%",
                },
                "balance": {
                    "score":  balance_result["test_score"],
                    "rating": balance_result["rating"],
                    "detail": balance_result["detail"],
                    "weight": "40%",
                },
            },
            "readiness_flags": readiness_flags,
            "what_was_measured": (
                "Three separate MediaPipe analyses combined: "
                "Jump test (explosive power + landing safety — 30%), "
                "Sprint test (acceleration mechanics + stride symmetry — 30%), "
                "Balance test (hip stability + knee alignment — 40%). "
                "Balance is weighted highest as it is most sensitive to fatigue and injury risk."
            ),
            "coach_action": decision,
        }

    finally:
        for path in tmp_paths:
            try: os.unlink(path)
            except OSError: pass

# =============================================================================
# END OF COACH ANALYSIS TOOLKIT
# =============================================================================
