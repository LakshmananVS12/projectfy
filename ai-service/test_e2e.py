"""
End-to-end validation script for the AI service.

Tests:
  1. POST /detect — photo detection endpoint
  2. POST /scan/video + GET /scan/{job_id} — async video job flow

Run this with the server already running: python test_e2e.py
"""
import json
import sys
import time

import cv2
import numpy as np
import requests

BASE_URL = "http://localhost:8000"


def create_test_image(path: str, w: int = 640, h: int = 480) -> str:
    """Create a synthetic road image with a dark rectangle simulating a pothole."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 180  # grey asphalt
    # Simulated pothole - dark irregular patch
    cv2.rectangle(img, (200, 200), (350, 320), (30, 25, 20), -1)
    cv2.ellipse(img, (275, 260), (70, 50), 0, 0, 360, (20, 15, 10), -1)
    # Simulated crack - thin dark lines
    cv2.line(img, (400, 100), (500, 400), (40, 35, 30), 2)
    cv2.line(img, (420, 120), (480, 380), (45, 40, 35), 1)
    cv2.imwrite(path, img)
    return path


def create_test_video(path: str, frames: int = 30, fps: int = 10) -> str:
    """
    Create a synthetic test video: a pothole-like dark rectangle moves 
    slightly across frames, simulating a dashcam driving past damage.
    """
    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for i in range(frames):
        img = np.ones((h, w, 3), dtype=np.uint8) * 180
        # Pothole moves slightly rightward across frames
        x_offset = i * 3
        cv2.rectangle(img, (200 + x_offset, 200), (350 + x_offset, 320), (30, 25, 20), -1)
        cv2.ellipse(img, (275 + x_offset, 260), (70, 50), 0, 0, 360, (20, 15, 10), -1)
        writer.write(img)

    writer.release()
    return path


def test_health():
    print("=" * 60)
    print("TEST: GET /health")
    r = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Response: {json.dumps(data, indent=2)}")
    assert r.status_code == 200
    assert data["status"] == "ok"
    print("  PASSED [OK]")
    return data


def test_photo_detect(image_path: str):
    print("=" * 60)
    print("TEST: POST /detect (photo)")
    with open(image_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/detect", files={"file": ("test.jpg", f, "image/jpeg")})

    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Detections: {len(data.get('detections', []))}")
    for det in data.get("detections", []):
        print(f"    - {det['class']}: conf={det['confidence']:.3f}, severity={det.get('severity', 'N/A')}, bbox={det['bbox']}")
    print(f"  Processing time: {data.get('processing_time_ms', 0):.1f} ms")
    print(f"  CNN heatmap present: {bool(data.get('cnn_heatmap_base64'))}")
    print(f"  ViT attention present: {bool(data.get('vit_attention_base64'))}")
    assert r.status_code == 200
    assert "detections" in data
    assert "cnn_heatmap_base64" in data
    assert "vit_attention_base64" in data
    assert "processing_time_ms" in data
    print("  PASSED [OK]")
    return data


def test_video_scan(video_path: str):
    print("=" * 60)
    print("TEST: POST /scan/video + GET /scan/{job_id}")

    # Submit job
    with open(video_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/scan/video", files={"file": ("test.mp4", f, "video/mp4")})

    print(f"  Submit status: {r.status_code}")
    data = r.json()
    job_id = data["job_id"]
    print(f"  Job ID: {job_id}")
    print(f"  Initial status: {data['status']}")
    assert r.status_code == 200
    assert data["status"] == "QUEUED"

    # Poll until complete (max 120 seconds)
    print("  Polling for completion...")
    for attempt in range(60):
        time.sleep(2)
        r = requests.get(f"{BASE_URL}/scan/{job_id}")
        poll_data = r.json()
        status = poll_data["status"]
        pct = poll_data.get("progress_pct", 0)
        print(f"    [{attempt+1}] status={status}, progress={pct}%")

        if status == "COMPLETED":
            results = poll_data.get("results", [])
            print(f"\n  COMPLETED — {len(results)} unique damage event(s)")
            for evt in results:
                print(f"    - Event {evt['event_id']}: {evt['damage_class']}, "
                      f"severity={evt['severity']}, conf={evt['confidence']:.3f}, "
                      f"timestamp={evt['timestamp_sec']:.1f}s")
                print(f"      representative_frame present: {bool(evt.get('representative_frame_base64'))}")
            print(f"\n  Deduplication check: {len(results)} unique events from 30-frame video")
            assert isinstance(results, list)
            print("  PASSED [OK]")
            return poll_data

        if status == "FAILED":
            print(f"  FAILED: {poll_data.get('error')}")
            sys.exit(1)

    print("  TIMEOUT — job did not complete in 120 seconds")
    sys.exit(1)


def test_404_job():
    print("=" * 60)
    print("TEST: GET /scan/nonexistent (should 404)")
    r = requests.get(f"{BASE_URL}/scan/nonexistent-job-id")
    print(f"  Status: {r.status_code}")
    assert r.status_code == 404
    print("  PASSED [OK]")


if __name__ == "__main__":
    print("\n>>> Road Damage Detection AI -- E2E Validation\n")

    test_health()

    img_path = create_test_image("_test_image.jpg")
    test_photo_detect(img_path)

    vid_path = create_test_video("_test_video.mp4")
    test_video_scan(vid_path)

    test_404_job()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED [OK]")
    print("=" * 60)
