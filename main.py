"""
GrassRoots Sports — AI Tracking Service
YOLOv8 + supervision ByteTracker player tracking microservice.

POST /track  — accepts a match video, returns per-player positions,
               heatmaps, distances, possession, and team stats.
GET  /health — liveness check for Render health monitor.
"""

from __future__ import annotations

import io
import os
import tempfile
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import supervision as sv
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="GrassRoots AI Tracker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://grassrootssports.live",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup — stays in memory across requests
_model: YOLO | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")  # smallest/fastest, auto-downloaded
    return _model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Real pitch dimensions (FIFA standard)
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Heatmap grid resolution (columns × rows)
HEATMAP_COLS = 20
HEATMAP_ROWS = 13

# Sample 1 frame per second — keeps processing time under 5 min for 90-min match
SAMPLE_FPS = 1

# Person class index in COCO (used by YOLOv8)
PERSON_CLASS_ID = 0

# ByteTracker parameters (tuned for football — players move fast, may be occluded)
TRACKER_CONFIG = sv.ByteTrackerArgs(
    track_activation_threshold=0.25,
    lost_track_buffer=50,        # keep lost track for 50 frames before dropping
    minimum_matching_threshold=0.8,
    frame_rate=SAMPLE_FPS,
    minimum_consecutive_frames=3,
)


# ---------------------------------------------------------------------------
# Jersey color extraction + team classification
# ---------------------------------------------------------------------------

def extract_jersey_color(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Extract the dominant HSV color from the top 50% of a player bounding box.
    Returns a 3-element array [H, S, V].
    """
    x1, y1, x2, y2 = map(int, box)

    # Clamp to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return np.array([0.0, 0.0, 0.0])

    # Top 50% = jersey area (avoid shorts/socks confusion)
    mid_y = y1 + (y2 - y1) // 2
    crop = frame[y1:mid_y, x1:x2]

    if crop.size == 0:
        return np.array([0.0, 0.0, 0.0])

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    if len(pixels) < 10:
        return pixels.mean(axis=0)

    # K-means with k=1 to get dominant color
    km = KMeans(n_clusters=1, n_init=3, random_state=42)
    km.fit(pixels)
    return km.cluster_centers_[0]


def classify_teams(
    tracker_ids: np.ndarray,
    boxes: np.ndarray,
    frame: np.ndarray,
    color_memory: dict[int, np.ndarray],
) -> dict[int, str]:
    """
    Classify each tracked player as 'home', 'away', or 'referee'.

    Strategy:
      1. Extract jersey HSV for each player (cache per tracker_id across frames).
      2. K-means with k=3 on all known colors.
      3. Largest two clusters = home + away. Smallest = referee.
      4. Return {tracker_id: team} mapping.
    """
    # Update color memory for players visible in this frame
    for tid, box in zip(tracker_ids, boxes):
        color = extract_jersey_color(frame, box)
        if tid not in color_memory:
            color_memory[tid] = color
        else:
            # Exponential moving average — smooth over time
            color_memory[tid] = 0.8 * color_memory[tid] + 0.2 * color

    if len(color_memory) < 3:
        # Not enough players visible yet — label all as 'home'
        return {tid: "home" for tid in tracker_ids}

    ids = list(color_memory.keys())
    colors = np.array(list(color_memory.values()), dtype=np.float32)

    k = min(3, len(ids))
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(colors)

    # Count players per cluster
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
    """
    Find the pitch bounding box by isolating green grass pixels (HSV).
    Returns (x_min, y_min, x_max, y_max). Falls back to full frame if detection fails.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green grass HSV range — tuned for typical football pitch footage
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological cleanup
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 1000:
        # Fallback: use full frame
        h, w = frame.shape[:2]
        return 0, 0, w, h

    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h


def pixel_to_pitch(
    px: float,
    py: float,
    pitch_bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    """
    Normalise pixel coordinates to pitch coordinates [0-1, 0-1].
    Returns (x_norm, y_norm) where (0,0) = top-left of pitch.
    """
    x_min, y_min, x_max, y_max = pitch_bounds
    pw = max(x_max - x_min, 1)
    ph = max(y_max - y_min, 1)

    x_norm = max(0.0, min(1.0, (px - x_min) / pw))
    y_norm = max(0.0, min(1.0, (py - y_min) / ph))
    return x_norm, y_norm


# ---------------------------------------------------------------------------
# Heatmap builder
# ---------------------------------------------------------------------------

def build_heatmap(positions: list[tuple[float, float]]) -> list[list[int]]:
    """
    Build a HEATMAP_ROWS × HEATMAP_COLS grid of presence counts
    from a list of (x_norm, y_norm) positions.
    """
    grid = [[0] * HEATMAP_COLS for _ in range(HEATMAP_ROWS)]
    for x_norm, y_norm in positions:
        col = min(int(x_norm * HEATMAP_COLS), HEATMAP_COLS - 1)
        row = min(int(y_norm * HEATMAP_ROWS), HEATMAP_ROWS - 1)
        grid[row][col] += 1
    return grid


# ---------------------------------------------------------------------------
# Distance calculator
# ---------------------------------------------------------------------------

def calculate_distance_m(positions: list[tuple[float, float]]) -> float:
    """
    Calculate total distance covered (metres) from normalised pitch positions.
    """
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(positions)):
        dx = (positions[i][0] - positions[i - 1][0]) * PITCH_LENGTH_M
        dy = (positions[i][1] - positions[i - 1][1]) * PITCH_WIDTH_M
        total += (dx**2 + dy**2) ** 0.5
    return round(total, 1)


# ---------------------------------------------------------------------------
# Main tracking endpoint
# ---------------------------------------------------------------------------

@app.post("/track")
async def track_video(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Accept a match video and return per-player tracking data.

    Response shape:
    {
      "players": [
        {
          "id": 1,
          "team": "home",
          "positions": [{"second": 0, "x": 0.45, "y": 0.3}, ...],
          "distance_m": 8420.5,
          "avg_x": 0.52,
          "avg_y": 0.48,
          "heatmap": [[0, 2, ...], ...]   // 13 rows × 20 cols
        }
      ],
      "stats": {
        "possession_home": 54,
        "possession_away": 46,
        "duration_seconds": 5400,
        "frames_processed": 5400
      },
      "video": {
        "width": 1920,
        "height": 1080,
        "fps": 25,
        "total_frames": 135000
      }
    }
    """
    # Validate mime type
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Write to temp file (OpenCV needs a real path)
    suffix = os.path.splitext(file.filename or "match.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        return _run_tracking(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _run_tracking(video_path: str) -> dict[str, Any]:
    model = get_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Cannot open video file")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample every N-th frame to achieve SAMPLE_FPS
    sample_every = max(1, int(round(original_fps / SAMPLE_FPS)))

    tracker = sv.ByteTracker(
        track_activation_threshold=TRACKER_CONFIG.track_activation_threshold,
        lost_track_buffer=TRACKER_CONFIG.lost_track_buffer,
        minimum_matching_threshold=TRACKER_CONFIG.minimum_matching_threshold,
        frame_rate=SAMPLE_FPS,
        minimum_consecutive_frames=TRACKER_CONFIG.minimum_consecutive_frames,
    )

    # Per-player data structures
    player_positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
    player_teams: dict[int, str] = {}
    player_seconds: dict[int, list[int]] = defaultdict(list)
    color_memory: dict[int, np.ndarray] = {}

    # Possession tracking — count frames each team controls the ball
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
            # Detect pitch bounds from the first 10 sampled frames (stabilises estimate)
            if frames_processed < 10:
                bounds = detect_pitch_bounds(frame)
                if pitch_bounds is None:
                    pitch_bounds = bounds
                else:
                    # Average with previous estimate
                    pitch_bounds = tuple(
                        int(0.7 * a + 0.3 * b)
                        for a, b in zip(pitch_bounds, bounds)
                    )  # type: ignore[assignment]

            if pitch_bounds is None:
                pitch_bounds = (0, 0, width, height)

            # Run YOLOv8 detection — only person class
            results = model(frame, classes=[PERSON_CLASS_ID], verbose=False)[0]

            # Convert to supervision Detections
            detections = sv.Detections.from_ultralytics(results)

            # Update tracker
            detections = tracker.update_with_detections(detections)

            if len(detections) > 0 and detections.tracker_id is not None:
                tracker_ids = detections.tracker_id
                boxes = detections.xyxy

                # Team classification
                team_map = classify_teams(tracker_ids, boxes, frame, color_memory)

                # Record positions for each player
                for tid, box in zip(tracker_ids, boxes):
                    # Use bottom-centre of bounding box as player position
                    px = (box[0] + box[2]) / 2.0
                    py = box[3]  # bottom of box = feet

                    x_norm, y_norm = pixel_to_pitch(px, py, pitch_bounds)
                    player_positions[int(tid)].append((x_norm, y_norm))
                    player_seconds[int(tid)].append(second)

                    team = team_map.get(int(tid), "home")
                    player_teams[int(tid)] = team

                # Simple possession heuristic:
                # Team with most players in central third owns possession that frame
                home_count = sum(
                    1 for tid in tracker_ids
                    if team_map.get(int(tid)) == "home"
                    and 0.33 < pixel_to_pitch(
                        (detections.xyxy[list(tracker_ids).index(tid)][0] +
                         detections.xyxy[list(tracker_ids).index(tid)][2]) / 2,
                        detections.xyxy[list(tracker_ids).index(tid)][3],
                        pitch_bounds,
                    )[0] < 0.67
                )
                away_count = sum(
                    1 for tid in tracker_ids
                    if team_map.get(int(tid)) == "away"
                    and 0.33 < pixel_to_pitch(
                        (detections.xyxy[list(tracker_ids).index(tid)][0] +
                         detections.xyxy[list(tracker_ids).index(tid)][2]) / 2,
                        detections.xyxy[list(tracker_ids).index(tid)][3],
                        pitch_bounds,
                    )[0] < 0.67
                )
                if home_count >= away_count:
                    possession_frames["home"] += 1
                else:
                    possession_frames["away"] += 1

            second += 1
            frames_processed += 1

        frame_idx += 1

    cap.release()

    # Build output per player
    players_out: list[dict[str, Any]] = []
    for tid, positions in player_positions.items():
        if len(positions) < 3:
            continue  # skip ghost detections

        seconds_list = player_seconds[tid]
        avg_x = round(sum(p[0] for p in positions) / len(positions), 3)
        avg_y = round(sum(p[1] for p in positions) / len(positions), 3)
        distance = calculate_distance_m(positions)
        heatmap = build_heatmap(positions)

        players_out.append(
            {
                "id": tid,
                "team": player_teams.get(tid, "home"),
                "positions": [
                    {"second": s, "x": round(x, 3), "y": round(y, 3)}
                    for s, (x, y) in zip(seconds_list, positions)
                ],
                "distance_m": distance,
                "avg_x": avg_x,
                "avg_y": avg_y,
                "heatmap": heatmap,
            }
        )

    # Possession percentages
    total_poss = possession_frames["home"] + possession_frames["away"]
    if total_poss > 0:
        poss_home = round(possession_frames["home"] / total_poss * 100)
        poss_away = 100 - poss_home
    else:
        poss_home, poss_away = 50, 50

    return {
        "players": players_out,
        "stats": {
            "possession_home": poss_home,
            "possession_away": poss_away,
            "duration_seconds": second,
            "frames_processed": frames_processed,
        },
        "video": {
            "width": width,
            "height": height,
            "fps": round(original_fps, 2),
            "total_frames": total_frames,
        },
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "grassroots-ai-tracker"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
