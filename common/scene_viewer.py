#!/usr/bin/env python3
"""
Interactive scene viewer with live detection overlays.

Shows camera feed with real-time object detection and scene interpretation.
Requires a display (run with X forwarding if remote).

Usage:
    python -m common.scene_viewer [options]

Keys:
    q - quit
    s - save current frame and scene description
    c - toggle color detection
    e - toggle edge/contour detection
    r - toggle relationship lines
    space - pause/resume
"""

import cv2
import numpy as np
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from .scene_interpreter import interpret_scene, annotate_scene, SceneDescription


class SceneViewer:
    """Interactive viewer with live scene interpretation."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        output_dir: str = "./captures",
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Detection settings
        self.use_color_detection = True
        self.use_contour_detection = True
        self.show_relationships = False
        self.min_area = 500

        # State
        self.paused = False
        self.last_frame = None
        self.last_scene: SceneDescription | None = None
        self.cap = None

        # Performance tracking
        self.fps_history = []
        self.last_time = time.time()

    def connect(self) -> bool:
        """Initialize camera."""
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {actual_w}x{actual_h}")

        return True

    def disconnect(self):
        """Release camera."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def save_capture(self):
        """Save current frame and scene description."""
        if self.last_frame is None:
            print("No frame to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw frame
        raw_path = self.output_dir / f"frame_{timestamp}.jpg"
        cv2.imwrite(str(raw_path), self.last_frame)

        # Save annotated frame
        if self.last_scene:
            annotated = annotate_scene(
                self.last_frame,
                self.last_scene,
                show_labels=True,
                show_relationships=self.show_relationships,
            )
            ann_path = self.output_dir / f"annotated_{timestamp}.jpg"
            cv2.imwrite(str(ann_path), annotated)

            # Save scene description
            json_path = self.output_dir / f"scene_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(self.last_scene.to_dict(), f, indent=2)

            print(f"Saved: {raw_path}, {ann_path}, {json_path}")
        else:
            print(f"Saved: {raw_path}")

    def draw_hud(self, frame: np.ndarray, scene: SceneDescription) -> np.ndarray:
        """Draw heads-up display with scene info."""
        result = frame.copy()
        height, width = result.shape[:2]

        # Semi-transparent overlay for text area
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, height - 60), (width, height), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.5, result, 0.5, 0)

        # Top HUD - detection status and FPS
        status_parts = []
        if self.use_color_detection:
            status_parts.append("Color:ON")
        else:
            status_parts.append("Color:OFF")

        if self.use_contour_detection:
            status_parts.append("Edge:ON")
        else:
            status_parts.append("Edge:OFF")

        if self.show_relationships:
            status_parts.append("Rel:ON")
        else:
            status_parts.append("Rel:OFF")

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / max(0.001, current_time - self.last_time)
        self.last_time = current_time
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        status_text = f"{' | '.join(status_parts)} | FPS: {avg_fps:.1f}"
        cv2.putText(
            result, status_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        # Object count and workspace state
        obj_text = f"Objects: {len(scene.objects)} | State: {scene.workspace_state}"
        if self.paused:
            obj_text += " | PAUSED"
        cv2.putText(
            result, obj_text, (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
        )

        # Bottom HUD - summary (truncated if too long)
        summary = scene.summary
        if len(summary) > 100:
            summary = summary[:97] + "..."
        cv2.putText(
            result, summary, (10, height - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Controls hint
        controls = "q:quit s:save c:color e:edge r:relations space:pause"
        cv2.putText(
            result, controls, (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1
        )

        return result

    def run(self):
        """Main viewer loop."""
        if not self.connect():
            return

        window_name = "Scene Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("Scene Viewer started. Press 'q' to quit.")
        print("Keys: s=save, c=color, e=edge, r=relations, space=pause")

        try:
            while True:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Error reading frame")
                        break

                    self.last_frame = frame

                    # Interpret scene
                    self.last_scene = interpret_scene(
                        frame,
                        use_color_detection=self.use_color_detection,
                        use_contour_detection=self.use_contour_detection,
                        min_area=self.min_area,
                    )
                else:
                    frame = self.last_frame

                if frame is None:
                    continue

                # Draw annotations
                display = annotate_scene(
                    frame,
                    self.last_scene,
                    show_labels=True,
                    show_relationships=self.show_relationships,
                )

                # Draw HUD
                display = self.draw_hud(display, self.last_scene)

                cv2.imshow(window_name, display)

                # Handle input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_capture()
                elif key == ord('c'):
                    self.use_color_detection = not self.use_color_detection
                    print(f"Color detection: {self.use_color_detection}")
                elif key == ord('e'):
                    self.use_contour_detection = not self.use_contour_detection
                    print(f"Contour detection: {self.use_contour_detection}")
                elif key == ord('r'):
                    self.show_relationships = not self.show_relationships
                    print(f"Show relationships: {self.show_relationships}")
                elif key == ord(' '):
                    self.paused = not self.paused
                    print(f"Paused: {self.paused}")

        finally:
            self.disconnect()
            cv2.destroyAllWindows()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive scene viewer with live detection"
    )
    parser.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "-W", "--width",
        type=int,
        default=1280,
        help="Camera width (default: 1280)"
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=720,
        help="Camera height (default: 720)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./captures",
        help="Output directory for saved captures (default: ./captures)"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum object area in pixels (default: 500)"
    )

    args = parser.parse_args()

    viewer = SceneViewer(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        output_dir=args.output,
    )
    viewer.min_area = args.min_area
    viewer.run()


if __name__ == "__main__":
    main()
