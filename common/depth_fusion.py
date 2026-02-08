"""
3D Scene Graph: YOLO + Depth Fusion.

Fuses Hailo YOLOv8 detections (USB camera, class labels) with PhotoNeo depth
pointcloud (accurate 3D positions) to produce labeled, 3D-positioned scene graphs.

Data flow:
  USB camera frame -> Hailo YOLO -> labeled 2D bboxes
  /tmp/phoxi_scan.npz -> transform pointcloud to robot frame -> BFS cluster -> 3D clusters
  Match YOLO detections <-> depth clusters by robot XY proximity
  -> SceneGraph3D {objects with label + 3D position + dimensions + color}
"""

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .vision import DetectedObject

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DEPTH_NPZ = "/tmp/phoxi_scan.npz"
DEFAULT_USB_CALIBRATION = "/tmp/aruco_calibration.npz"
DEFAULT_DEPTH_CALIBRATION = "/tmp/depth_calibration.npz"

# Clustering params
GRID_CELL_M = 0.02  # 2cm grid cells (finer = better block separation)
MIN_CLUSTER_POINTS = 20  # minimum points to form a cluster
Z_ABOVE_DESK_M = 0.03  # points must be above this height (robot frame, ~1.7cm above desk)

# Workspace bounds for filtering (robot frame, meters)
WORKSPACE_X = (-0.4, 0.75)
WORKSPACE_Y = (-0.5, 0.5)
WORKSPACE_Z = (0.01, 0.5)

# Matching threshold
MATCH_THRESHOLD_M = 0.10  # max XY distance for YOLO-depth matching


@dataclass
class DepthCluster:
    """A cluster of 3D points from the depth camera, in robot frame."""
    center_m: tuple[float, float, float]  # (x, y, z) centroid in robot frame
    z_max_m: float  # highest point in cluster
    point_count: int
    bounds_m: dict  # {x_min, x_max, y_min, y_max, z_min, z_max}
    dimensions_m: tuple[float, float, float]  # (dx, dy, dz) extent

    def to_dict(self) -> dict:
        return {
            "center_m": {"x": round(self.center_m[0], 4), "y": round(self.center_m[1], 4), "z": round(self.center_m[2], 4)},
            "z_max_m": round(self.z_max_m, 4),
            "point_count": self.point_count,
            "dimensions_m": {"x": round(self.dimensions_m[0], 4), "y": round(self.dimensions_m[1], 4), "z": round(self.dimensions_m[2], 4)},
        }


@dataclass
class Object3D:
    """A detected object with 3D position and class label."""
    label: str
    confidence: float
    position_m: tuple[float, float, float]  # (x, y, z) in robot frame
    dimensions_m: tuple[float, float, float]  # (dx, dy, dz) extent
    color_name: str | None = None
    source: str = "fused"  # "fused", "yolo_only", "depth_only"
    grasp_width_m: float | None = None  # min(dx, dy) for gripper_grasp
    point_count: int = 0

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "position_m": {
                "x": round(self.position_m[0], 4),
                "y": round(self.position_m[1], 4),
                "z": round(self.position_m[2], 4),
            },
            "dimensions_m": {
                "x": round(self.dimensions_m[0], 4),
                "y": round(self.dimensions_m[1], 4),
                "z": round(self.dimensions_m[2], 4),
            },
            "source": self.source,
        }
        if self.color_name:
            d["color"] = self.color_name
        if self.grasp_width_m is not None:
            d["grasp_width_m"] = round(self.grasp_width_m, 4)
        if self.point_count > 0:
            d["point_count"] = self.point_count
        return d


@dataclass
class SceneGraph3D:
    """Complete 3D scene graph with labeled, positioned objects."""
    objects: list[Object3D] = field(default_factory=list)
    summary: str = ""

    @property
    def total_objects(self) -> int:
        return len(self.objects)

    @property
    def fused_count(self) -> int:
        return sum(1 for o in self.objects if o.source == "fused")

    @property
    def yolo_only_count(self) -> int:
        return sum(1 for o in self.objects if o.source == "yolo_only")

    @property
    def depth_only_count(self) -> int:
        return sum(1 for o in self.objects if o.source == "depth_only")

    def to_dict(self) -> dict:
        return {
            "total_objects": self.total_objects,
            "fused_count": self.fused_count,
            "yolo_only_count": self.yolo_only_count,
            "depth_only_count": self.depth_only_count,
            "objects": [o.to_dict() for o in self.objects],
            "summary": self.summary,
        }


def cluster_pointcloud(
    pointcloud: np.ndarray,
    depth: np.ndarray,
    transform: np.ndarray,
    grid_cell_m: float = GRID_CELL_M,
    min_points: int = MIN_CLUSTER_POINTS,
    z_threshold_m: float = Z_ABOVE_DESK_M,
) -> list[DepthCluster]:
    """
    Cluster elevated points in the depth pointcloud using grid-based BFS.

    Args:
        pointcloud: (H, W, 3) array in camera frame (mm)
        depth: (H, W) depth array (mm), 0 = invalid
        transform: 4x4 SE(3) matrix from camera to robot frame
        grid_cell_m: Grid cell size for clustering
        min_points: Minimum points per cluster
        z_threshold_m: Minimum z in robot frame (above desk)

    Returns:
        List of DepthCluster sorted by point count (descending)
    """
    h, w = depth.shape

    # Flatten valid points and transform to robot frame
    valid_mask = depth > 0
    valid_indices = np.argwhere(valid_mask)  # (N, 2) of [row, col]
    if len(valid_indices) == 0:
        return []

    # Get camera-frame points in meters
    pts_cam_mm = pointcloud[valid_mask]  # (N, 3)
    pts_cam_m = pts_cam_mm.astype(np.float64) / 1000.0

    # Transform to robot frame: p_robot = R @ p_cam + t
    R = transform[:3, :3]
    t = transform[:3, 3]
    pts_robot = (R @ pts_cam_m.T).T + t  # (N, 3)

    # Filter: above desk and within workspace
    mask = (
        (pts_robot[:, 2] > z_threshold_m) &
        (pts_robot[:, 0] > WORKSPACE_X[0]) & (pts_robot[:, 0] < WORKSPACE_X[1]) &
        (pts_robot[:, 1] > WORKSPACE_Y[0]) & (pts_robot[:, 1] < WORKSPACE_Y[1]) &
        (pts_robot[:, 2] < WORKSPACE_Z[1])
    )

    elevated_pts = pts_robot[mask]  # (M, 3)
    if len(elevated_pts) < min_points:
        return []

    # Grid-based BFS clustering (no scipy needed)
    # Quantize XY to grid cells
    gx = ((elevated_pts[:, 0] - elevated_pts[:, 0].min()) / grid_cell_m).astype(int)
    gy = ((elevated_pts[:, 1] - elevated_pts[:, 1].min()) / grid_cell_m).astype(int)

    # Build grid -> point indices mapping
    grid_to_points: dict[tuple[int, int], list[int]] = {}
    for i in range(len(elevated_pts)):
        key = (int(gx[i]), int(gy[i]))
        if key not in grid_to_points:
            grid_to_points[key] = []
        grid_to_points[key].append(i)

    # BFS connected components on grid
    visited_cells: set[tuple[int, int]] = set()
    clusters: list[list[int]] = []

    for cell in grid_to_points:
        if cell in visited_cells:
            continue

        # BFS from this cell
        queue = deque([cell])
        visited_cells.add(cell)
        component_points: list[int] = []

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) in grid_to_points:
                component_points.extend(grid_to_points[(cx, cy)])

            # Check 8-connected neighbors
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (cx + dx, cy + dy)
                    if neighbor not in visited_cells and neighbor in grid_to_points:
                        visited_cells.add(neighbor)
                        queue.append(neighbor)

        if len(component_points) >= min_points:
            clusters.append(component_points)

    # Build DepthCluster objects
    result = []
    for point_indices in clusters:
        pts = elevated_pts[point_indices]
        center = pts.mean(axis=0)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        dims = maxs - mins

        result.append(DepthCluster(
            center_m=(float(center[0]), float(center[1]), float(center[2])),
            z_max_m=float(maxs[2]),
            point_count=len(point_indices),
            bounds_m={
                "x_min": float(mins[0]), "x_max": float(maxs[0]),
                "y_min": float(mins[1]), "y_max": float(maxs[1]),
                "z_min": float(mins[2]), "z_max": float(maxs[2]),
            },
            dimensions_m=(float(dims[0]), float(dims[1]), float(dims[2])),
        ))

    # Filter out non-object clusters
    filtered = []
    for c in result:
        dx, dy, dz = c.dimensions_m
        # Robot arm: very large cluster with high z
        if c.point_count > 10000 and c.z_max_m > 0.10:
            logger.debug(f"Filtering robot arm cluster: {c.point_count}pts, z_max={c.z_max_m:.3f}")
            continue
        # Behind workspace
        if c.center_m[0] < 0.15:
            logger.debug(f"Filtering cluster behind workspace: x={c.center_m[0]:.3f}")
            continue
        # Flat surface artifacts (tape, paper edges, marker corners): z-height < 1.5cm
        if dz < 0.015:
            logger.debug(f"Filtering flat cluster: dz={dz:.3f}m at ({c.center_m[0]:.3f}, {c.center_m[1]:.3f})")
            continue
        # Oversized in XY: > 12cm in any horizontal axis (not a single graspable object)
        if max(dx, dy) > 0.12:
            logger.debug(f"Filtering oversized cluster: {dx:.3f}x{dy:.3f}m, {c.point_count}pts")
            continue
        filtered.append(c)

    # Sort by point count descending
    filtered.sort(key=lambda c: c.point_count, reverse=True)
    return filtered


def _identify_color(bgr: tuple[int, int, int] | None) -> str | None:
    """Identify color name from BGR values."""
    if bgr is None:
        return None
    pixel = np.uint8([[bgr]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv
    if s < 30:
        if v < 50:
            return "black"
        elif v > 200:
            return "white"
        else:
            return "gray"
    if h < 10 or h > 160:
        return "red"
    elif h < 22:
        return "orange"
    elif h < 38:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 130:
        return "blue"
    elif h < 160:
        return "purple"
    return None


def match_yolo_to_depth(
    yolo_detections: list[DetectedObject],
    clusters: list[DepthCluster],
    usb_homography: np.ndarray,
    table_z: float = 0.013,
    threshold_m: float = MATCH_THRESHOLD_M,
) -> list[Object3D]:
    """
    Match YOLO 2D detections to depth 3D clusters by robot XY proximity.

    Args:
        yolo_detections: YOLO DetectedObjects with bbox + label
        clusters: 3D depth clusters in robot frame
        usb_homography: 3x3 homography matrix (pixel -> robot XY)
        table_z: Table height in robot frame (for yolo_only fallback)
        threshold_m: Max XY distance for matching

    Returns:
        List of Object3D (fused, yolo_only, and depth_only)
    """
    objects: list[Object3D] = []

    # Convert YOLO bbox centers to robot XY via homography
    yolo_robot_xy = []
    for det in yolo_detections:
        cx, cy = det.bbox.center
        pt = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pt, usb_homography)
        rx, ry = float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
        yolo_robot_xy.append((rx, ry))

    # Build distance matrix
    n_yolo = len(yolo_detections)
    n_depth = len(clusters)

    if n_yolo > 0 and n_depth > 0:
        distances = np.zeros((n_yolo, n_depth))
        for i in range(n_yolo):
            for j in range(n_depth):
                dx = yolo_robot_xy[i][0] - clusters[j].center_m[0]
                dy = yolo_robot_xy[i][1] - clusters[j].center_m[1]
                distances[i, j] = np.sqrt(dx * dx + dy * dy)

        # Greedy nearest-neighbor matching
        matched_yolo: set[int] = set()
        matched_depth: set[int] = set()

        # Sort all pairs by distance
        pairs = []
        for i in range(n_yolo):
            for j in range(n_depth):
                pairs.append((distances[i, j], i, j))
        pairs.sort()

        for dist, yi, di in pairs:
            if dist > threshold_m:
                break
            if yi in matched_yolo or di in matched_depth:
                continue

            # Fused match
            det = yolo_detections[yi]
            cluster = clusters[di]
            grasp_w = min(cluster.dimensions_m[0], cluster.dimensions_m[1])

            objects.append(Object3D(
                label=det.label,
                confidence=det.confidence,
                position_m=cluster.center_m,
                dimensions_m=cluster.dimensions_m,
                color_name=_identify_color(det.color_bgr),
                source="fused",
                grasp_width_m=grasp_w,
                point_count=cluster.point_count,
            ))

            matched_yolo.add(yi)
            matched_depth.add(di)
    else:
        matched_yolo = set()
        matched_depth = set()

    # Unmatched YOLO -> yolo_only (position from homography, z=table)
    for i, det in enumerate(yolo_detections):
        if i in matched_yolo:
            continue
        rx, ry = yolo_robot_xy[i]
        objects.append(Object3D(
            label=det.label,
            confidence=det.confidence,
            position_m=(rx, ry, table_z),
            dimensions_m=(0.0, 0.0, 0.0),
            color_name=_identify_color(det.color_bgr),
            source="yolo_only",
        ))

    # Unmatched depth clusters -> depth_only with "unknown" label
    for j, cluster in enumerate(clusters):
        if j in matched_depth:
            continue
        grasp_w = min(cluster.dimensions_m[0], cluster.dimensions_m[1])
        objects.append(Object3D(
            label="unknown",
            confidence=0.0,
            position_m=cluster.center_m,
            dimensions_m=cluster.dimensions_m,
            source="depth_only",
            grasp_width_m=grasp_w,
            point_count=cluster.point_count,
        ))

    return objects


def _generate_summary(objects: list[Object3D]) -> str:
    """Generate natural language summary of the 3D scene."""
    if not objects:
        return "No objects detected on the table."

    n = len(objects)

    # Group by label
    label_counts: dict[str, int] = {}
    for obj in objects:
        label_counts[obj.label] = label_counts.get(obj.label, 0) + 1

    label_strs = []
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        if count == 1:
            label_strs.append(label)
        else:
            label_strs.append(f"{count} {label}s")

    parts = [f"{n} object{'s' if n != 1 else ''}: {', '.join(label_strs)}."]

    # Describe positions relative to robot
    for obj in objects:
        x, y, z = obj.position_m
        # Left/right from robot's perspective
        if y < -0.1:
            lr = "right"
        elif y > 0.1:
            lr = "left"
        else:
            lr = "center"
        # Near/far
        if x < 0.35:
            nf = "near"
        elif x > 0.50:
            nf = "far"
        else:
            nf = ""

        pos_str = f"{nf} {lr}".strip() if nf else lr
        color_str = f"{obj.color_name} " if obj.color_name else ""
        parts.append(f"{color_str}{obj.label} at {pos_str} (x={x:.2f}, y={y:.2f}, z={z:.3f})")

    return " | ".join(parts)


def build_scene_graph(
    frame: np.ndarray,
    depth_npz_path: str = DEFAULT_DEPTH_NPZ,
    usb_calibration_path: str = DEFAULT_USB_CALIBRATION,
    depth_calibration_path: str = DEFAULT_DEPTH_CALIBRATION,
    confidence_threshold: float = 0.3,
    max_age_seconds: float = 60.0,
) -> SceneGraph3D:
    """
    Build a 3D scene graph by fusing YOLO detections with depth pointcloud.

    Args:
        frame: BGR image from USB camera
        depth_npz_path: Path to saved depth scan NPZ
        usb_calibration_path: Path to USB camera ArUco calibration
        depth_calibration_path: Path to depth camera SE(3) calibration
        confidence_threshold: YOLO confidence threshold
        max_age_seconds: Reject depth scans older than this

    Returns:
        SceneGraph3D with fused objects
    """
    # Validate depth NPZ exists and isn't stale
    npz_path = Path(depth_npz_path)
    if not npz_path.exists():
        return SceneGraph3D(summary=f"No depth scan found at {depth_npz_path}. Call capture_depth + save_scan first.")

    age = time.time() - npz_path.stat().st_mtime
    if age > max_age_seconds:
        return SceneGraph3D(summary=f"Depth scan is {age:.0f}s old (max {max_age_seconds:.0f}s). Capture a fresh scan.")

    # Load calibrations
    usb_cal_path = Path(usb_calibration_path)
    if not usb_cal_path.exists():
        return SceneGraph3D(summary=f"USB camera calibration not found at {usb_calibration_path}")

    depth_cal_path = Path(depth_calibration_path)
    if not depth_cal_path.exists():
        return SceneGraph3D(summary=f"Depth calibration not found at {depth_calibration_path}")

    usb_cal = np.load(usb_calibration_path, allow_pickle=True)
    homography = usb_cal["H"]
    table_z = float(usb_cal.get("table_z", 0.013))

    depth_cal = np.load(depth_calibration_path)
    transform = depth_cal["transform"]  # 4x4 SE(3)

    # Load depth scan
    scan = np.load(depth_npz_path)
    pointcloud = scan["pointcloud"]
    depth = scan["depth"]

    # Run YOLO detection on USB camera frame
    from .hailo_detector import get_hailo_detector
    detector = get_hailo_detector()
    yolo_detections = []
    if detector is not None:
        yolo_detections = detector.detect(frame, confidence_threshold=confidence_threshold)
        logger.info(f"YOLO detected {len(yolo_detections)} objects")
    else:
        logger.warning("Hailo YOLO detector not available, using depth-only mode")

    # Cluster depth pointcloud
    clusters = cluster_pointcloud(pointcloud, depth, transform)
    logger.info(f"Found {len(clusters)} depth clusters")

    # Match and fuse
    objects = match_yolo_to_depth(
        yolo_detections, clusters, homography,
        table_z=table_z,
    )

    # Generate summary
    summary = _generate_summary(objects)

    return SceneGraph3D(objects=objects, summary=summary)
