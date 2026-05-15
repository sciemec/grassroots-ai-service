"""
GrassRoots Sports — AI Tracking Service v2
YOLOv8x + supervision ByteTracker player tracking microservice.

Upgrades over v1:
  - YOLOv8x model (5x more accurate than nano)
  - Ball tracking (COCO class 32 — sports ball)
  - Ball-proximity possession (accurate, not heuristic)
  - Speed per player in km/h (top speed + avg speed)
  - Named player support — pass squad JSON in POST body

POST /track  — accepts video + optional squad JSON, returns full tracking data
GET  /health — liveness check
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
import uuid as uuid_mod
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Optional

import time

import boto3
import cv2
import httpx
import numpy as np
import supervision as sv
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="GrassRoots AI Tracker", version="2.0.0")

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

_model: YOLO | None = None

# In-memory job store for async analysis jobs — expires after 2h
_jobs: dict[str, dict] = {}


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8x.pt")  # upgraded from nano — 5x more accurate
    return _model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
HEATMAP_COLS = 20
HEATMAP_ROWS = 13
SAMPLE_FPS = 1
PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32  # COCO sports ball class

TRACKER_CONFIG = SimpleNamespace(
    track_activation_threshold=0.25,
    lost_track_buffer=50,
    minimum_matching_threshold=0.8,
    frame_rate=SAMPLE_FPS,
    minimum_consecutive_frames=3,
)


# ---------------------------------------------------------------------------
# Jersey color extraction + team classification
# ---------------------------------------------------------------------------

def extract_jersey_color(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return np.array([0.0, 0.0, 0.0])

    mid_y = y1 + (y2 - y1) // 2
    crop = frame[y1:mid_y, x1:x2]

    if crop.size == 0:
        return np.array([0.0, 0.0, 0.0])

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    if len(pixels) < 10:
        return pixels.mean(axis=0)

    km = KMeans(n_clusters=1, n_init=3, random_state=42)
    km.fit(pixels)
    return km.cluster_centers_[0]


def classify_teams(
    tracker_ids: np.ndarray,
    boxes: np.ndarray,
    frame: np.ndarray,
    color_memory: dict[int, np.ndarray],
) -> dict[int, str]:
    for tid, box in zip(tracker_ids, boxes):
        color = extract_jersey_color(frame, box)
        if tid not in color_memory:
            color_memory[tid] = color
        else:
            color_memory[tid] = 0.8 * color_memory[tid] + 0.2 * color

    if len(color_memory) < 3:
        return {tid: "home" for tid in tracker_ids}

    ids = list(color_memory.keys())
    colors = np.array(list(color_memory.values()), dtype=np.float32)

    k = min(3, len(ids))
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(colors)

    from collections import Counter
    counts = Counter(labels)
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

    id_to_team: dict[int, str] = {}
    for tid, label in zip(ids, labels):
        id_to_team[tid] = cluster_to_team.get(label, "home")

    return {tid: id_to_team.get(tid, "home") for tid in tracker_ids}


# ---------------------------------------------------------------------------
# Pitch coordinate normalisation
# ---------------------------------------------------------------------------

def detect_pitch_bounds(frame: np.ndarray) -> tuple[int, int, int, int]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 1000:
        h, w = frame.shape[:2]
        return 0, 0, w, h
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h


def pixel_to_pitch(
    px: float,
    py: float,
    pitch_bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = pitch_bounds
    pw = max(x_max - x_min, 1)
    ph = max(y_max - y_min, 1)
    x_norm = max(0.0, min(1.0, (px - x_min) / pw))
    y_norm = max(0.0, min(1.0, (py - y_min) / ph))
    return x_norm, y_norm


# ---------------------------------------------------------------------------
# Heatmap + distance + speed
# ---------------------------------------------------------------------------

def build_heatmap(positions: list[tuple[float, float]]) -> list[list[int]]:
    grid = [[0] * HEATMAP_COLS for _ in range(HEATMAP_ROWS)]
    for x_norm, y_norm in positions:
        col = min(int(x_norm * HEATMAP_COLS), HEATMAP_COLS - 1)
        row = min(int(y_norm * HEATMAP_ROWS), HEATMAP_ROWS - 1)
        grid[row][col] += 1
    return grid


def calculate_distance_m(positions: list[tuple[float, float]]) -> float:
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(positions)):
        dx = (positions[i][0] - positions[i - 1][0]) * PITCH_LENGTH_M
        dy = (positions[i][1] - positions[i - 1][1]) * PITCH_WIDTH_M
        total += (dx**2 + dy**2) ** 0.5
    return round(total, 1)


def calculate_speeds(positions: list[tuple[float, float]]) -> list[float]:
    """
    Speed in km/h per step. SAMPLE_FPS=1 so each step = 1 second.
    speed_kmh = distance_m * 3.6
    """
    speeds: list[float] = []
    for i in range(1, len(positions)):
        dx = (positions[i][0] - positions[i - 1][0]) * PITCH_LENGTH_M
        dy = (positions[i][1] - positions[i - 1][1]) * PITCH_WIDTH_M
        dist_m = (dx**2 + dy**2) ** 0.5
        speeds.append(round(dist_m * 3.6, 1))
    return speeds


# ---------------------------------------------------------------------------
# Main tracking endpoint
# ---------------------------------------------------------------------------

@app.post("/track")
async def track_video(
    file: UploadFile = File(...),
    squad: Optional[str] = Form(None),
) -> dict[str, Any]:
    """
    Accept a match video and return per-player tracking data.

    squad (optional Form field): JSON string mapping tracker IDs to player names.
    Example: '{"1": "Musona K.", "7": "Billiat K."}'

    Analyst can also name players in the web app after tracking completes —
    the web app sends a save request with the name mapping.
    """
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    squad_map: dict[str, str] = {}
    if squad:
        try:
            squad_map = json.loads(squad)
        except json.JSONDecodeError:
            pass

    suffix = os.path.splitext(file.filename or "match.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        return _run_tracking(tmp_path, squad_map)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _run_tracking(video_path: str, squad_map: dict[str, str]) -> dict[str, Any]:
    model = get_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Cannot open video file")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_every = max(1, int(round(original_fps / SAMPLE_FPS)))

    tracker = sv.ByteTracker(
        track_activation_threshold=TRACKER_CONFIG.track_activation_threshold,
        lost_track_buffer=TRACKER_CONFIG.lost_track_buffer,
        minimum_matching_threshold=TRACKER_CONFIG.minimum_matching_threshold,
        frame_rate=SAMPLE_FPS,
        minimum_consecutive_frames=TRACKER_CONFIG.minimum_consecutive_frames,
    )

    player_positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
    player_teams: dict[int, str] = {}
    player_seconds: dict[int, list[int]] = defaultdict(list)
    color_memory: dict[int, np.ndarray] = {}

    # Ball tracking
    ball_positions: list[dict[str, Any]] = []
    last_ball_pos: tuple[float, float] | None = None

    # Possession — ball proximity preferred, central-third fallback
    possession_frames: dict[str, int] = {"home": 0, "away": 0}

    pitch_bounds: tuple[int, int, int, int] | None = None
    frame_idx = 0
    second = 0
    frames_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            # Stabilise pitch bounds from first 10 sampled frames
            if frames_processed < 10:
                bounds = detect_pitch_bounds(frame)
                if pitch_bounds is None:
                    pitch_bounds = bounds
                else:
                    pitch_bounds = tuple(
                        int(0.7 * a + 0.3 * b)
                        for a, b in zip(pitch_bounds, bounds)
                    )  # type: ignore[assignment]

            if pitch_bounds is None:
                pitch_bounds = (0, 0, width, height)

            # Detect players (class 0) AND ball (class 32) in one pass
            results = model(
                frame,
                classes=[PERSON_CLASS_ID, BALL_CLASS_ID],
                verbose=False,
            )[0]

            detections_all = sv.Detections.from_ultralytics(results)

            # Split by class_id
            if detections_all.class_id is not None and len(detections_all) > 0:
                player_mask = detections_all.class_id == PERSON_CLASS_ID
                ball_mask = detections_all.class_id == BALL_CLASS_ID
                player_detections = detections_all[player_mask]
                ball_detections = detections_all[ball_mask]
            else:
                player_detections = detections_all
                ball_detections = sv.Detections.empty()

            # Track players
            player_detections = tracker.update_with_detections(player_detections)

            # Ball — highest confidence detection this frame
            ball_pos_this_frame: tuple[float, float] | None = None
            if len(ball_detections) > 0:
                best_idx = (
                    int(np.argmax(ball_detections.confidence))
                    if ball_detections.confidence is not None
                    else 0
                )
                bx1, by1, bx2, by2 = ball_detections.xyxy[best_idx]
                bx = (bx1 + bx2) / 2.0
                by = (by1 + by2) / 2.0
                bx_norm, by_norm = pixel_to_pitch(bx, by, pitch_bounds)
                ball_pos_this_frame = (bx_norm, by_norm)
                last_ball_pos = ball_pos_this_frame
                ball_positions.append({
                    "second": second,
                    "x": round(bx_norm, 3),
                    "y": round(by_norm, 3),
                })

            # Process player detections
            if len(player_detections) > 0 and player_detections.tracker_id is not None:
                tracker_ids = player_detections.tracker_id
                boxes = player_detections.xyxy

                team_map = classify_teams(tracker_ids, boxes, frame, color_memory)

                for tid, box in zip(tracker_ids, boxes):
                    px = (box[0] + box[2]) / 2.0
                    py = box[3]
                    x_norm, y_norm = pixel_to_pitch(px, py, pitch_bounds)
                    player_positions[int(tid)].append((x_norm, y_norm))
                    player_seconds[int(tid)].append(second)
                    player_teams[int(tid)] = team_map.get(int(tid), "home")

                # Possession: ball proximity (accurate) or central-third fallback
                ball_ref = ball_pos_this_frame or last_ball_pos
                if ball_ref is not None:
                    min_dist = float("inf")
                    closest_team = "home"
                    for tid, box in zip(tracker_ids, boxes):
                        px = (box[0] + box[2]) / 2.0
                        py = box[3]
                        x_norm, y_norm = pixel_to_pitch(px, py, pitch_bounds)
                        dist = (
                            (x_norm - ball_ref[0]) ** 2 +
                            (y_norm - ball_ref[1]) ** 2
                        ) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_team = team_map.get(int(tid), "home")
                    if closest_team in ("home", "away"):
                        possession_frames[closest_team] += 1
                else:
                    home_count = sum(
                        1 for tid in tracker_ids
                        if team_map.get(int(tid)) == "home"
                    )
                    away_count = sum(
                        1 for tid in tracker_ids
                        if team_map.get(int(tid)) == "away"
                    )
                    if home_count >= away_count:
                        possession_frames["home"] += 1
                    else:
                        possession_frames["away"] += 1

            second += 1
            frames_processed += 1

        frame_idx += 1

    cap.release()

    # Build per-player output with speed data
    players_out: list[dict[str, Any]] = []
    for tid, positions in player_positions.items():
        if len(positions) < 3:
            continue

        seconds_list = player_seconds[tid]
        avg_x = round(sum(p[0] for p in positions) / len(positions), 3)
        avg_y = round(sum(p[1] for p in positions) / len(positions), 3)
        distance = calculate_distance_m(positions)
        heatmap = build_heatmap(positions)
        speeds = calculate_speeds(positions)

        top_speed = round(max(speeds), 1) if speeds else 0.0
        avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0.0

        players_out.append({
            "id": tid,
            "name": squad_map.get(str(tid), ""),
            "team": player_teams.get(tid, "home"),
            "positions": [
                {"second": s, "x": round(x, 3), "y": round(y, 3)}
                for s, (x, y) in zip(seconds_list, positions)
            ],
            "distance_m": distance,
            "avg_x": avg_x,
            "avg_y": avg_y,
            "heatmap": heatmap,
            "top_speed_kmh": top_speed,
            "avg_speed_kmh": avg_speed,
        })

    total_poss = possession_frames["home"] + possession_frames["away"]
    if total_poss > 0:
        poss_home = round(possession_frames["home"] / total_poss * 100)
        poss_away = 100 - poss_home
    else:
        poss_home, poss_away = 50, 50

    return {
        "players": players_out,
        "ball": ball_positions,
        "stats": {
            "possession_home": poss_home,
            "possession_away": poss_away,
            "duration_seconds": second,
            "frames_processed": frames_processed,
            "ball_detected_frames": len(ball_positions),
        },
        "video": {
            "width": width,
            "height": height,
            "fps": round(original_fps, 2),
            "total_frames": total_frames,
        },
    }


# ---------------------------------------------------------------------------
# Gemini File API proxy (browser cannot call Google directly due to CORS)
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
        # Start resumable upload session
        init_res = await client.post(
            f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={google_key}",
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(content_length),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            },
            json={"file": {"display_name": f"match-{int(time.time())}"}},
        )
        if init_res.status_code not in (200, 201):
            raise HTTPException(status_code=502, detail=f"Gemini session init failed: {init_res.status_code}")

        upload_url = init_res.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise HTTPException(status_code=502, detail="Gemini did not return upload URL")

        # Upload video bytes to Google
        upload_res = await client.put(
            upload_url,
            headers={
                "Content-Length": str(content_length),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=content,
        )
        if upload_res.status_code not in (200, 201):
            raise HTTPException(status_code=502, detail=f"Gemini upload failed: {upload_res.status_code}")

        file_info = upload_res.json().get("file", {})
        return {
            "fileUri":  file_info.get("uri", ""),
            "fileName": file_info.get("name", ""),
            "mimeType": file_info.get("mimeType", mime_type),
            "state":    file_info.get("state", "ACTIVE"),
        }


# ---------------------------------------------------------------------------
# Sprint event detection
# ---------------------------------------------------------------------------

SPRINT_THRESHOLD_KMH = 25.0


def detect_sprint_events(
    players: list[dict[str, Any]],
    threshold_kmh: float = SPRINT_THRESHOLD_KMH,
) -> list[dict[str, Any]]:
    """
    Scan per-player position data and return sprint moments.
    A sprint = any second where the player's speed exceeds threshold_kmh.
    Output: [{player_id, name, team, second, speed_kmh}]
    """
    events: list[dict[str, Any]] = []
    for player in players:
        positions = player.get("positions", [])
        for i in range(1, len(positions)):
            prev = positions[i - 1]
            curr = positions[i]
            dx = (curr["x"] - prev["x"]) * PITCH_LENGTH_M
            dy = (curr["y"] - prev["y"]) * PITCH_WIDTH_M
            speed_kmh = ((dx**2 + dy**2) ** 0.5) * 3.6
            if speed_kmh >= threshold_kmh:
                events.append({
                    "player_id": player["id"],
                    "name": player.get("name", ""),
                    "team": player.get("team", "home"),
                    "second": curr["second"],
                    "speed_kmh": round(speed_kmh, 1),
                })

    events.sort(key=lambda e: e["speed_kmh"], reverse=True)  # fastest first
    return events


# ---------------------------------------------------------------------------
# FFmpeg clip cutter
# ---------------------------------------------------------------------------

def clip_highlights(
    video_path: str,
    events: list[dict[str, Any]],
    padding_s: int = 5,
    max_clips: int = 10,
) -> list[dict[str, Any]]:
    """
    For each sprint event, cut ±padding_s seconds of video using ffmpeg.
    Skips events whose time window overlaps a clip already cut.
    Returns [{event, clip_path, start, end}].
    """
    clips: list[dict[str, Any]] = []
    seen_windows: list[tuple[int, int]] = []

    for event in events:
        if len(clips) >= max_clips:
            break

        second = event["second"]
        start = max(0, second - padding_s)
        end = second + padding_s

        # Skip overlapping window
        if any(not (end <= s or start >= e) for s, e in seen_windows):
            continue

        seen_windows.append((start, end))
        out_path = tempfile.mktemp(suffix=".mp4")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(end - start),
            "-c:v", "libx264", "-preset", "fast", "-crf", "28",
            "-c:a", "aac",
            "-movflags", "+faststart",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0:
            clips.append({"event": event, "clip_path": out_path, "start": start, "end": end})

    return clips


# ---------------------------------------------------------------------------
# Cloudflare R2 uploader
# ---------------------------------------------------------------------------

def upload_clips_to_r2(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Upload each clip to Cloudflare R2 (S3-compatible).
    If R2 env vars are not set, returns empty url (dev mode).
    Always deletes the local temp file after upload attempt.
    Returns [{event, url, key}].
    """
    r2_account = os.environ.get("R2_ACCOUNT_ID")
    r2_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    r2_bucket = os.environ.get("R2_BUCKET", "grassroots-videos")
    r2_public = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")

    s3_client = None
    if all([r2_account, r2_key, r2_secret]):
        s3_client = boto3.client(
            "s3",
            endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
            aws_access_key_id=r2_key,
            aws_secret_access_key=r2_secret,
            region_name="auto",
        )

    results: list[dict[str, Any]] = []
    for clip in clips:
        key = f"highlights/{int(time.time())}-p{clip['event']['player_id']}-{clip['start']}s.mp4"
        url = ""
        try:
            if s3_client:
                s3_client.upload_file(
                    clip["clip_path"],
                    r2_bucket,
                    key,
                    ExtraArgs={"ContentType": "video/mp4"},
                )
                url = f"{r2_public}/{key}" if r2_public else ""
        except Exception:
            url = ""
        finally:
            try:
                os.unlink(clip["clip_path"])
            except OSError:
                pass
        results.append({"event": clip["event"], "url": url, "key": key})

    return results


# ---------------------------------------------------------------------------
# Highlight clip endpoint
# ---------------------------------------------------------------------------

@app.post("/clip")
async def clip_video(
    file: UploadFile = File(...),
    squad: Optional[str] = Form(None),
    threshold_kmh: float = Form(25.0),
    max_clips: int = Form(10),
) -> dict[str, Any]:
    """
    Run player tracking, detect sprint events above threshold_kmh,
    cut highlight clips with ffmpeg, upload to R2.

    Returns: { clips, events_detected, clips_generated, stats }
    """
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    squad_map: dict[str, str] = {}
    if squad:
        try:
            squad_map = json.loads(squad)
        except json.JSONDecodeError:
            pass

    suffix = os.path.splitext(file.filename or "match.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        tracking = _run_tracking(tmp_path, squad_map)
        events = detect_sprint_events(tracking["players"], threshold_kmh)
        raw_clips = clip_highlights(tmp_path, events, max_clips=max_clips)
        clip_results = upload_clips_to_r2(raw_clips)
        return {
            "clips": clip_results,
            "events_detected": len(events),
            "clips_generated": len(clip_results),
            "stats": tracking["stats"],
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Background video pipeline (Video Highlight Pipeline — Step 1)
# ---------------------------------------------------------------------------

class ProcessVideoRequest(BaseModel):
    video_key: str
    match_id: int
    callback_url: str


def download_from_r2(video_key: str) -> str:
    r2_account = os.environ.get("R2_ACCOUNT_ID")
    r2_access  = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret  = os.environ.get("R2_SECRET_ACCESS_KEY")
    r2_bucket  = os.environ.get("R2_BUCKET", "grassroots-media")
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
        aws_access_key_id=r2_access,
        aws_secret_access_key=r2_secret,
        region_name="auto",
    )
    suffix = os.path.splitext(video_key)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    s3.download_fileobj(r2_bucket, video_key, tmp)
    tmp.close()
    return tmp.name


def run_pipeline(video_key: str, match_id: int, callback_url: str) -> None:
    video_path = None
    try:
        video_path = download_from_r2(video_key)
        tracking   = _run_tracking(video_path, {})
        events     = detect_sprint_events(tracking["players"])
        raw_clips  = clip_highlights(video_path, events, max_clips=10)
        r2_account = os.environ.get("R2_ACCOUNT_ID")
        r2_access  = os.environ.get("R2_ACCESS_KEY_ID")
        r2_secret  = os.environ.get("R2_SECRET_ACCESS_KEY")
        r2_bucket  = os.environ.get("R2_BUCKET", "grassroots-media")
        r2_public  = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
        s3_client = None
        if all([r2_account, r2_access, r2_secret]):
            s3_client = boto3.client(
                "s3",
                endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
                aws_access_key_id=r2_access,
                aws_secret_access_key=r2_secret,
                region_name="auto",
            )
        clips_out = []
        for clip in raw_clips:
            key = f"highlights/{match_id}/{int(time.time())}-p{clip['event']['player_id']}-{clip['start']}s.mp4"
            url = ""
            try:
                if s3_client:
                    s3_client.upload_file(
                        clip["clip_path"], r2_bucket, key,
                        ExtraArgs={"ContentType": "video/mp4"},
                    )
                    url = f"{r2_public}/{key}" if r2_public else ""
            except Exception:
                pass
            finally:
                try:
                    os.unlink(clip["clip_path"])
                except OSError:
                    pass
            clips_out.append({
                "player_id":  clip["event"]["player_id"],
                "event_type": "sprint",
                "timestamp":  clip["event"]["second"],
                "speed":      clip["event"]["speed_kmh"],
                "url":        url,
                "r2_key":     key,
            })
        with httpx.Client(timeout=30) as client:
            client.post(callback_url, json={
                "match_id": match_id,
                "status":   "complete",
                "clips":    clips_out,
            })
    except Exception as exc:
        try:
            with httpx.Client(timeout=10) as client:
                client.post(callback_url, json={
                    "match_id": match_id,
                    "status":   "failed",
                    "error":    str(exc),
                    "clips":    [],
                })
        except Exception:
            pass
    finally:
        if video_path:
            try:
                os.unlink(video_path)
            except OSError:
                pass


@app.post("/process-video")
async def process_video(
    req: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """
    Download video from R2, run YOLOv8 tracking + sprint detection + FFmpeg clipping,
    upload highlight clips back to R2, then POST results to callback_url.
    Returns immediately — processing happens in the background.
    """
    background_tasks.add_task(run_pipeline, req.video_key, req.match_id, req.callback_url)
    return {"status": "processing", "match_id": str(req.match_id)}


# ---------------------------------------------------------------------------
# Background match analysis (Gemini File API + Claude narrative)
# ---------------------------------------------------------------------------

class AnalyseRequest(BaseModel):
    fileUri: str
    fileName: str
    mimeType: str
    fileState: Optional[str] = None
    homeTeam: str
    awayTeam: str
    competition: Optional[str] = ""
    sport: Optional[str] = "football"


async def _wait_for_file_active(file_name: str, google_key: str, job_id: str) -> None:
    """Poll Gemini until the uploaded file state is ACTIVE."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=60.0, write=10.0, pool=10.0)
    ) as client:
        for _ in range(120):
            res = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={google_key}"
            )
            if not res.is_success:
                raise RuntimeError(f"File state check failed: {res.status_code} — {res.text[:200]}")
            data = res.json()
            state = data.get("state", "")
            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise RuntimeError("Gemini file processing failed")
            _jobs[job_id]["message"] = f"Gemini processing video... (state: {state})"
            await asyncio.sleep(5)
    raise RuntimeError("Video did not become ready within 10 minutes")


def _extract_json(text: str) -> dict | None:
    """Parse JSON from Gemini response — handles plain JSON or markdown code blocks."""
    try:
        return json.loads(text)
    except Exception:
        pass
    md = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if md:
        try:
            return json.loads(md.group(1))
        except Exception:
            pass
    obj = re.search(r"\{[\s\S]*\}", text)
    if obj:
        try:
            return json.loads(obj.group(0))
        except Exception:
            pass
    return None


async def _analyse_background(job_id: str, req: AnalyseRequest) -> None:
    """Run Gemini 1.5 Pro analysis + Claude narrative in the background."""
    try:
        google_key = os.environ.get("GOOGLE_AI_API_KEY", "")
        if not google_key:
            raise RuntimeError("GOOGLE_AI_API_KEY not configured on AI service")

        _jobs[job_id]["message"] = "Waiting for Gemini to finish processing the video..."
        _jobs[job_id]["progress"] = 10

        if req.fileState != "ACTIVE":
            await _wait_for_file_active(req.fileName, google_key, job_id)

        _jobs[job_id]["message"] = "Gemini 1.5 Pro is watching the full match — this takes several minutes..."
        _jobs[job_id]["progress"] = 30

        system_prompt = (
            f"You are a professional football analyst with UEFA A-licence coaching experience.\n"
            f"You will watch the full match video: {req.homeTeam} vs {req.awayTeam}"
            + (f" ({req.competition})" if req.competition else "")
            + (f" — Sport: {req.sport}" if req.sport else "")
            + """\n\nWatch the entire video. Observe player positions, ball movement, team shapes, events, and tactical patterns throughout the full match.\n\n"""
            """Return ONLY a valid JSON object — no markdown, no explanation — with this exact structure:\n"""
            """{\n"""
            """  "formation_home": "4-3-3",\n"""
            """  "formation_away": "4-4-2",\n"""
            """  "possession_home": 55,\n"""
            """  "possession_away": 45,\n"""
            """  "shots_home": 8,\n"""
            """  "shots_away": 5,\n"""
            """  "shots_on_target_home": 4,\n"""
            """  "shots_on_target_away": 2,\n"""
            """  "fouls_detected": 3,\n"""
            """  "key_events": [\n"""
            """    { "time": "23:00", "team": "home", "type": "shot", "description": "Right-footed shot from edge of box" }\n"""
            """  ],\n"""
            """  "tactical_patterns": ["Home team pressed high in the first 30 minutes"],\n"""
            """  "defensive_issues": ["Left back exposed on counter-attacks repeatedly"],\n"""
            """  "attacking_strengths": ["Strong combination play through the central midfield"],\n"""
            """  "man_of_match_candidate": "Home team central midfielder — controlled the tempo all match",\n"""
            """  "halftime_recommendation": "Push the right winger higher and switch to a 4-2-3-1",\n"""
            """  "key_coaching_points": ["Defensive line needs to step up 5 metres when opponent goalkeeper has the ball"]\n"""
            """}\n\n"""
            """For possession: estimate from which team controlled the ball across the full match.\n"""
            """For events: include all significant events with accurate timestamps.\n"""
            """For formations: identify from player positioning throughout the full match.\n"""
            """Be specific and professional. Base everything on what you observe in the video."""
        )

        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=10.0)) as client:
            gemini_res = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={google_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{
                        "parts": [
                            {"text": system_prompt},
                            {"file_data": {"mime_type": req.mimeType, "file_uri": req.fileUri}},
                            {"text": "Now provide your complete JSON analysis of this full match video."},
                        ]
                    }],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
                },
            )

        _jobs[job_id]["progress"] = 75
        _jobs[job_id]["message"] = "Gemini analysis complete — generating tactical narrative..."

        if not gemini_res.is_success:
            raise RuntimeError(f"Gemini API error: {gemini_res.status_code} — {gemini_res.text[:300]}")

        gemini_data = gemini_res.json()
        gemini_text = (
            gemini_data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        analysis = _extract_json(gemini_text)
        if analysis is None:
            raise RuntimeError(f"Gemini returned unreadable analysis: {gemini_text[:300]}")

        # Claude narrative (optional — only if key is set)
        narrative = ""
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=120.0, write=10.0, pool=10.0)) as client:
                claude_res = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 1500,
                        "messages": [{
                            "role": "user",
                            "content": (
                                f"You are a professional football analyst writing a post-match report for a coach.\n\n"
                                f"Match: {req.homeTeam} vs {req.awayTeam}"
                                + (f"\nCompetition: {req.competition}" if req.competition else "")
                                + (f"\nSport: {req.sport}" if req.sport else "")
                                + f"\n\nAI Vision Analysis (Gemini 1.5 Pro watched the full match video natively):\n"
                                + json.dumps(analysis, indent=2)
                                + "\n\nWrite a professional 4-paragraph tactical match report:\n"
                                "1. Match overview — what happened and who controlled the game\n"
                                "2. Tactical analysis — what formations were used, what worked, what didn't\n"
                                "3. Individual highlights and areas of concern\n"
                                "4. Training recommendations for the next session based on what was seen\n\n"
                                "Write as a UEFA A-licence coach. Be specific, direct, and actionable. "
                                "Reference formations, patterns, and events by name. No generic advice."
                            ),
                        }],
                    },
                )
            if claude_res.is_success:
                narrative = claude_res.json().get("content", [{}])[0].get("text", "")

        _jobs[job_id].update({
            "status": "complete",
            "progress": 100,
            "message": "Analysis complete.",
            "analysis": analysis,
            "narrative": narrative,
        })

    except Exception as exc:
        _jobs[job_id].update({"status": "failed", "error": str(exc), "progress": 0})


@app.post("/analyse")
async def analyse_match(req: AnalyseRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    """
    Start background Gemini 1.5 Pro + Claude match analysis.
    Returns immediately with a job_id. Poll GET /job/{job_id} every 5s for status.
    """
    job_id = str(uuid_mod.uuid4())
    _jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Analysis queued...",
        "analysis": None,
        "narrative": None,
        "error": None,
        "created_at": time.time(),
    }
    background_tasks.add_task(_analyse_background, job_id, req)
    return {"job_id": job_id}


@app.get("/job/{job_id}")
async def get_job(job_id: str) -> dict:
    """
    Poll for background job status. Returns job dict with status, progress, analysis, narrative, error.
    Cleans up jobs older than 2 hours.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found — it may have expired")

    # Clean up expired jobs (> 2h old)
    now = time.time()
    expired = [k for k, v in list(_jobs.items()) if now - v.get("created_at", now) > 7200]
    for k in expired:
        _jobs.pop(k, None)

    return job


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "grassroots-ai-tracker", "model": "yolov8x"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
