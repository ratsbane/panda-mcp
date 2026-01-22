"""
Data collection for visuomotor policy training.

Collects paired (image, robot_state, camera_pose) data for training
a low-level control policy. Supports variable camera positions via
ArUco marker-based camera pose estimation.
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime


@dataclass
class CameraPose:
    """Camera pose in world coordinates (relative to ArUco markers)."""
    position: tuple[float, float, float]  # x, y, z in meters
    rotation: tuple[float, float, float]  # roll, pitch, yaw in radians
    confidence: float  # 0-1, based on marker detection quality
    markers_detected: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RobotState:
    """Complete robot state at a moment in time."""
    joint_positions: list[float]  # 7 joint angles in radians
    end_effector_position: tuple[float, float, float]  # x, y, z in meters
    end_effector_orientation: tuple[float, float, float]  # roll, pitch, yaw
    gripper_width: float  # meters
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataSample:
    """A single training sample."""
    image_path: str  # Relative path to image file
    robot_state: RobotState
    camera_pose: Optional[CameraPose]
    session_id: str
    sample_id: int
    timestamp: float

    def to_dict(self) -> dict:
        d = {
            "image_path": self.image_path,
            "robot_state": self.robot_state.to_dict(),
            "camera_pose": self.camera_pose.to_dict() if self.camera_pose else None,
            "session_id": self.session_id,
            "sample_id": self.sample_id,
            "timestamp": self.timestamp,
        }
        return d


class ArUcoTracker:
    """
    Track camera pose using ArUco markers.

    Place ArUco markers at known positions in the workspace.
    The tracker detects them and estimates camera pose.
    """

    def __init__(
        self,
        marker_size: float = 0.05,  # 5cm markers
        dictionary_id: int = cv2.aruco.DICT_4X4_50,
    ):
        self.marker_size = marker_size
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

        # Known marker positions in world frame (to be configured)
        # Format: {marker_id: (x, y, z, rx, ry, rz)}
        self.marker_positions: dict[int, tuple] = {}

        # Camera intrinsics (default values, should be calibrated)
        self.camera_matrix = np.array([
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros(5, dtype=np.float32)

    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """Set camera intrinsic parameters."""
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def add_marker(self, marker_id: int, position: tuple[float, float, float],
                   rotation: tuple[float, float, float] = (0, 0, 0)):
        """Register a marker at a known world position."""
        self.marker_positions[marker_id] = (*position, *rotation)

    def detect_markers(self, image: np.ndarray) -> dict[int, np.ndarray]:
        """Detect ArUco markers in image, return corners by ID."""
        corners, ids, rejected = self.detector.detectMarkers(image)

        detected = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                detected[marker_id] = corners[i]

        return detected

    def estimate_camera_pose(self, image: np.ndarray) -> Optional[CameraPose]:
        """
        Estimate camera pose from detected ArUco markers.

        Returns camera position/orientation in world frame.
        """
        detected = self.detect_markers(image)

        if not detected:
            return None

        # Find markers with known positions
        known_markers = {
            mid: corners for mid, corners in detected.items()
            if mid in self.marker_positions
        }

        if not known_markers:
            return CameraPose(
                position=(0, 0, 0),
                rotation=(0, 0, 0),
                confidence=0.0,
                markers_detected=len(detected)
            )

        # Estimate pose from each known marker and average
        positions = []
        rotations = []

        for marker_id, corners in known_markers.items():
            # Estimate pose of marker relative to camera
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )

            # Get marker's known world position
            marker_world = self.marker_positions[marker_id]
            mx, my, mz = marker_world[:3]

            # Convert rotation vector to matrix
            R_marker_to_cam, _ = cv2.Rodrigues(rvec[0])

            # Camera position relative to marker
            cam_pos_marker_frame = -R_marker_to_cam.T @ tvec[0].T

            # Transform to world frame (simplified - assumes markers are axis-aligned)
            cam_pos_world = (
                mx + cam_pos_marker_frame[0, 0],
                my + cam_pos_marker_frame[1, 0],
                mz + cam_pos_marker_frame[2, 0],
            )
            positions.append(cam_pos_world)

            # Extract rotation (simplified)
            rot_vec, _ = cv2.Rodrigues(R_marker_to_cam.T)
            rotations.append(rot_vec.flatten())

        # Average positions and rotations
        avg_pos = tuple(np.mean(positions, axis=0))
        avg_rot = tuple(np.mean(rotations, axis=0))
        confidence = min(1.0, len(known_markers) / 3)  # More markers = more confident

        return CameraPose(
            position=avg_pos,
            rotation=avg_rot,
            confidence=confidence,
            markers_detected=len(detected)
        )

    def draw_markers(self, image: np.ndarray) -> np.ndarray:
        """Draw detected markers on image for visualization."""
        result = image.copy()
        detected = self.detect_markers(image)

        if detected:
            corners_list = list(detected.values())
            ids = np.array(list(detected.keys())).reshape(-1, 1)
            cv2.aruco.drawDetectedMarkers(result, corners_list, ids)

            # Draw pose axes for known markers
            for marker_id, corners in detected.items():
                if marker_id in self.marker_positions:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                    )
                    cv2.drawFrameAxes(
                        result, self.camera_matrix, self.dist_coeffs,
                        rvec[0], tvec[0], self.marker_size * 0.5
                    )

        return result


class DataCollector:
    """
    Collects training data for visuomotor policy.

    Records synchronized (image, robot_state, camera_pose) tuples.
    """

    def __init__(
        self,
        output_dir: str = "./training_data",
        session_name: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.session_id = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / self.session_id

        # Create directories
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "images").mkdir(exist_ok=True)

        # Tracking
        self.samples: list[DataSample] = []
        self.sample_count = 0

        # ArUco tracker for camera pose
        self.aruco_tracker = ArUcoTracker()

        # Session metadata
        self.metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "camera_intrinsics": None,
            "marker_positions": {},
            "notes": "",
        }

    def configure_camera(self, fx: float, fy: float, cx: float, cy: float):
        """Set camera intrinsic parameters."""
        self.aruco_tracker.set_camera_intrinsics(fx, fy, cx, cy)
        self.metadata["camera_intrinsics"] = {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy
        }

    def add_marker(self, marker_id: int, position: tuple[float, float, float],
                   rotation: tuple[float, float, float] = (0, 0, 0)):
        """Register an ArUco marker at a known position."""
        self.aruco_tracker.add_marker(marker_id, position, rotation)
        self.metadata["marker_positions"][marker_id] = {
            "position": position,
            "rotation": rotation,
        }

    def record_sample(
        self,
        image: np.ndarray,
        robot_state: RobotState,
        estimate_camera_pose: bool = True,
    ) -> DataSample:
        """
        Record a single training sample.

        Args:
            image: BGR image from camera
            robot_state: Current robot state
            estimate_camera_pose: Whether to estimate camera pose from ArUco

        Returns:
            The recorded DataSample
        """
        # Save image
        image_filename = f"frame_{self.sample_count:06d}.jpg"
        image_path = self.session_dir / "images" / image_filename
        cv2.imwrite(str(image_path), image)

        # Estimate camera pose if requested
        camera_pose = None
        if estimate_camera_pose:
            camera_pose = self.aruco_tracker.estimate_camera_pose(image)

        # Create sample
        sample = DataSample(
            image_path=f"images/{image_filename}",
            robot_state=robot_state,
            camera_pose=camera_pose,
            session_id=self.session_id,
            sample_id=self.sample_count,
            timestamp=time.time(),
        )

        self.samples.append(sample)
        self.sample_count += 1

        return sample

    def save_session(self):
        """Save session metadata and samples to disk."""
        # Update metadata
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["total_samples"] = self.sample_count

        # Save metadata
        with open(self.session_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Save samples
        samples_data = [s.to_dict() for s in self.samples]
        with open(self.session_dir / "samples.json", "w") as f:
            json.dump(samples_data, f, indent=2)

        print(f"Session saved: {self.session_dir}")
        print(f"Total samples: {self.sample_count}")

    def add_note(self, note: str):
        """Add a note to the session metadata."""
        self.metadata["notes"] += f"\n{datetime.now().isoformat()}: {note}"


def create_robot_state_from_status(status: dict) -> RobotState:
    """Create RobotState from MCP get_status response."""
    ee = status["end_effector"]
    return RobotState(
        joint_positions=status["joint_positions_rad"],
        end_effector_position=(
            ee["position_m"]["x"],
            ee["position_m"]["y"],
            ee["position_m"]["z"],
        ),
        end_effector_orientation=(
            ee["orientation_rad"]["roll"],
            ee["orientation_rad"]["pitch"],
            ee["orientation_rad"]["yaw"],
        ),
        gripper_width=status["gripper_width_m"],
        timestamp=time.time(),
    )


def generate_aruco_markers(output_dir: str = "./aruco_markers", count: int = 10):
    """Generate printable ArUco marker images."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    for i in range(count):
        marker = cv2.aruco.generateImageMarker(dictionary, i, 200)
        cv2.imwrite(str(output_path / f"marker_{i}.png"), marker)

    print(f"Generated {count} ArUco markers in {output_path}")
    print("Print these at 5cm x 5cm and place in the workspace")


# CLI for generating markers
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data collection utilities")
    parser.add_argument("--generate-markers", action="store_true",
                        help="Generate printable ArUco markers")
    parser.add_argument("--marker-count", type=int, default=10,
                        help="Number of markers to generate")
    parser.add_argument("--output-dir", type=str, default="./aruco_markers",
                        help="Output directory for markers")

    args = parser.parse_args()

    if args.generate_markers:
        generate_aruco_markers(args.output_dir, args.marker_count)
