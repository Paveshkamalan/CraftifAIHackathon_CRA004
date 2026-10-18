"""
Script to generate demo/sample_video.mp4 and demo/high_load_video.mp4
for AdaPipe benchmarks and runtime adaptation demonstrations.
"""

import math
from pathlib import Path
import cv2
import numpy as np


def generate_entry_video(
    output_path: str,
    num_frames: int = 1500,  # 50 seconds at 30 fps
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    high_load: bool = False,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    num_persons = 12 if high_load else 4
    persons = []
    for i in range(num_persons):
        speed = (3.0 + (i % 5) * 1.5) * (1.8 if high_load else 1.0)
        direction = 1 if (i % 2 == 0) else -1
        x_init = (i * (width // num_persons)) if direction == 1 else (width - (i * (width // num_persons)))
        y_pos = int(height * (0.45 + 0.04 * (i % 7)))
        scale = 0.8 + 0.1 * (i % 4)
        persons.append({
            "x": float(x_init),
            "y": float(y_pos),
            "speed": speed * direction,
            "scale": scale,
            "id": i + 1,
            "color": (40 + (i * 35) % 180, 50 + (i * 45) % 180, 200 - (i * 20) % 150)
        })

    for f in range(num_frames):
        # Draw background: doorway / security entry corridor
        frame = np.full((height, width, 3), 235, dtype=np.uint8)

        # Floor with perspective
        pts_floor = np.array([[0, int(height * 0.6)], [width, int(height * 0.6)], [width, height], [0, height]])
        cv2.fillPoly(frame, [pts_floor], (190, 195, 200))

        # Entry door frame
        cv2.rectangle(frame, (int(width * 0.35), int(height * 0.2)), (int(width * 0.65), int(height * 0.8)), (70, 80, 90), 8)
        cv2.putText(frame, "BUILDING ENTRANCE - ZONE A", (int(width * 0.36), int(height * 0.17)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        # Time overlay
        timestamp_str = f"T: {f / fps:05.2f}s | Frame {f:04d} | Mode: {'HIGH-LOAD' if high_load else 'STANDARD'}"
        cv2.putText(frame, timestamp_str, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

        # Render moving person figures
        for p in persons:
            p["x"] += p["speed"]
            if p["x"] > width + 80:
                p["x"] = -80
            elif p["x"] < -80:
                p["x"] = width + 80

            px = int(p["x"])
            py = int(p["y"])
            pw = int(48 * p["scale"])
            ph = int(120 * p["scale"])

            # Head
            head_r = int(14 * p["scale"])
            cv2.circle(frame, (px, py - ph // 2 - head_r), head_r, p["color"], -1)
            # Torso
            cv2.rectangle(frame, (px - pw // 2, py - ph // 2), (px + pw // 2, py + ph // 4), p["color"], -1)
            # Legs
            cv2.line(frame, (px - pw // 4, py + ph // 4), (px - pw // 3, py + ph // 2), p["color"], int(8 * p["scale"]))
            cv2.line(frame, (px + pw // 4, py + ph // 4), (px + pw // 3, py + ph // 2), p["color"], int(8 * p["scale"]))

            # Person label tag for visibility
            cv2.putText(frame, f"person #{p['id']}", (px - pw // 2, py - ph // 2 - head_r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 1)

        out.write(frame)

    out.release()
    print(f"Generated {output_path}: {num_frames} frames ({num_frames / fps:.1f}s), {width}x{height} @ {fps}fps")


if __name__ == "__main__":
    generate_entry_video("demo/sample_video.mp4", num_frames=1500, fps=30, high_load=False)
    generate_entry_video("demo/high_load_video.mp4", num_frames=1500, fps=30, high_load=True)
