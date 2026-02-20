"""
Persistent embedding database for spatial memory.

Stores (DINOv2 embedding, robot coordinates, metadata) tuples in SQLite.
Supports nearest-neighbor lookup for pixel-to-robot coordinate mapping.

The embeddings are computed by a remote DINOv2 server (running on Spark GPU).
The database file lives on the Pi's filesystem and survives crashes.

Usage:
    db = SpatialDB()
    db.connect()

    # Store a calibration point
    embedding = db.embed_image(jpeg_bytes)
    db.add_point(embedding, robot_xyz=(0.45, 0.08, 0.025),
                 pixel_xy=(600, 450), metadata={"color": "red"})

    # Query: given a new image, find nearest robot coordinates
    query_emb = db.embed_image(new_jpeg_bytes)
    matches = db.query(query_emb, k=3)
"""

import io
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "spatial.db"
DEFAULT_EMBED_URL = "http://spark:8091"


@dataclass
class SpatialPoint:
    """A stored point in the spatial database."""
    id: int
    embedding: np.ndarray
    robot_x: float
    robot_y: float
    robot_z: float
    pixel_x: Optional[int]
    pixel_y: Optional[int]
    timestamp: float
    metadata: dict


class SpatialDB:
    """Persistent spatial memory backed by SQLite + DINOv2 embeddings."""

    def __init__(self, db_path: str = None, embed_url: str = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.embed_url = embed_url or DEFAULT_EMBED_URL
        self.conn = None
        self._embed_dim = None

    def connect(self):
        """Open (or create) the database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")  # crash-safe
        self._create_tables()
        count = self.count()
        logger.info(f"SpatialDB opened at {self.db_path} ({count} points)")

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                embed_dim INTEGER NOT NULL,
                robot_x REAL NOT NULL,
                robot_y REAL NOT NULL,
                robot_z REAL NOT NULL,
                pixel_x INTEGER,
                pixel_y INTEGER,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        self.conn.commit()

    def count(self) -> int:
        """Number of points in the database."""
        row = self.conn.execute("SELECT COUNT(*) FROM points").fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Embedding computation (via remote DINOv2 server)
    # ------------------------------------------------------------------

    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        """Send image to Spark DINOv2 server, get embedding back."""
        import urllib.request

        req = urllib.request.Request(
            f"{self.embed_url}/embed",
            data=image_bytes,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        embedding = np.array(result["embedding"], dtype=np.float32)
        self._embed_dim = len(embedding)
        logger.debug(f"Embedded image: dim={len(embedding)}, "
                     f"time={result.get('time_ms', '?')}ms")
        return embedding

    def embed_crop(self, image_bytes: bytes,
                   x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Send image + crop region to Spark, get embedding of the crop."""
        import urllib.request

        metadata = json.dumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        body = metadata.encode() + b"\n" + image_bytes

        req = urllib.request.Request(
            f"{self.embed_url}/embed_crop",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        embedding = np.array(result["embedding"], dtype=np.float32)
        self._embed_dim = len(embedding)
        return embedding

    def _resize_for_embed(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to ~256px before sending over network.
        DINOv2 resizes to 256 then center-crops to 224 anyway,
        so sending full 1280x720 wastes bandwidth."""
        h, w = frame.shape[:2]
        if max(h, w) > 320:
            scale = 320 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        return frame

    def embed_frame(self, frame: np.ndarray) -> np.ndarray:
        """Embed an OpenCV frame (BGR numpy array)."""
        small = self._resize_for_embed(frame)
        _, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return self.embed_image(jpeg.tobytes())

    def embed_frame_crop(self, frame: np.ndarray,
                         x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Embed a crop from an OpenCV frame."""
        crop = frame[y1:y2, x1:x2]
        small = self._resize_for_embed(crop)
        _, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return self.embed_image(jpeg.tobytes())

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add_point(self, embedding: np.ndarray,
                  robot_xyz: tuple,
                  pixel_xy: tuple = None,
                  metadata: dict = None) -> int:
        """Store a point. Returns the row ID."""
        meta_str = json.dumps(metadata or {})
        px = pixel_xy[0] if pixel_xy else None
        py = pixel_xy[1] if pixel_xy else None

        cursor = self.conn.execute(
            """INSERT INTO points
               (embedding, embed_dim, robot_x, robot_y, robot_z,
                pixel_x, pixel_y, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (embedding.tobytes(), len(embedding),
             robot_xyz[0], robot_xyz[1], robot_xyz[2],
             px, py, time.time(), meta_str),
        )
        self.conn.commit()
        logger.info(f"Added point #{cursor.lastrowid}: "
                    f"robot=({robot_xyz[0]:.3f}, {robot_xyz[1]:.3f}, {robot_xyz[2]:.3f})")
        return cursor.lastrowid

    def get_all_embeddings(self) -> tuple:
        """Load all embeddings and metadata. Returns (ids, embeddings, points)."""
        rows = self.conn.execute(
            """SELECT id, embedding, embed_dim, robot_x, robot_y, robot_z,
                      pixel_x, pixel_y, timestamp, metadata
               FROM points ORDER BY id"""
        ).fetchall()

        if not rows:
            return np.array([]), np.array([]).reshape(0, 0), []

        ids = []
        embeddings = []
        points = []

        for row in rows:
            rid, emb_blob, edim, rx, ry, rz, px, py, ts, meta = row
            emb = np.frombuffer(emb_blob, dtype=np.float32).copy()
            ids.append(rid)
            embeddings.append(emb)
            points.append(SpatialPoint(
                id=rid, embedding=emb,
                robot_x=rx, robot_y=ry, robot_z=rz,
                pixel_x=px, pixel_y=py,
                timestamp=ts, metadata=json.loads(meta),
            ))

        return (np.array(ids),
                np.vstack(embeddings),
                points)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, embedding: np.ndarray, k: int = 3) -> list:
        """
        Find the k nearest points by cosine similarity.
        Returns list of (similarity, SpatialPoint) tuples, highest first.
        """
        ids, all_embs, points = self.get_all_embeddings()
        if len(points) == 0:
            return []

        # Normalize for cosine similarity
        query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        emb_norms = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)

        similarities = emb_norms @ query_norm
        top_k = np.argsort(similarities)[::-1][:k]

        return [(float(similarities[i]), points[i]) for i in top_k]

    def query_robot_coords(self, embedding: np.ndarray, k: int = 3) -> tuple:
        """
        Query and return interpolated robot coordinates.
        Uses distance-weighted average of k nearest neighbors.
        Returns (robot_x, robot_y, robot_z, confidence).
        """
        matches = self.query(embedding, k=k)
        if not matches:
            return None

        # Distance-weighted interpolation
        weights = []
        coords = []
        for sim, point in matches:
            w = max(sim, 0.0) ** 2  # square similarity for sharper weighting
            weights.append(w)
            coords.append((point.robot_x, point.robot_y, point.robot_z))

        total_w = sum(weights)
        if total_w < 1e-8:
            # All similarities near zero — just use closest
            p = matches[0][1]
            return (p.robot_x, p.robot_y, p.robot_z, 0.0)

        rx = sum(w * c[0] for w, c in zip(weights, coords)) / total_w
        ry = sum(w * c[1] for w, c in zip(weights, coords)) / total_w
        rz = sum(w * c[2] for w, c in zip(weights, coords)) / total_w
        confidence = matches[0][0]  # similarity of best match

        return (rx, ry, rz, confidence)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_point(self, point_id: int):
        """Remove a point by ID."""
        self.conn.execute("DELETE FROM points WHERE id = ?", (point_id,))
        self.conn.commit()

    def clear(self):
        """Remove all points."""
        self.conn.execute("DELETE FROM points")
        self.conn.commit()
        logger.info("Database cleared")

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def stats(self) -> dict:
        """Database statistics."""
        count = self.count()
        if count == 0:
            return {"count": 0}

        row = self.conn.execute(
            """SELECT MIN(robot_x), MAX(robot_x),
                      MIN(robot_y), MAX(robot_y),
                      MIN(robot_z), MAX(robot_z),
                      MIN(timestamp), MAX(timestamp)
               FROM points"""
        ).fetchone()

        return {
            "count": count,
            "x_range": (row[0], row[1]),
            "y_range": (row[2], row[3]),
            "z_range": (row[4], row[5]),
            "time_range": (row[6], row[7]),
            "db_path": str(self.db_path),
        }
