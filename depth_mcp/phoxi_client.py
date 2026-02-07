"""SSH client for PhotoNeo phoxi_grab tool on tuppy."""

import asyncio
import logging
import struct
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Remote paths on tuppy
PHOXI_GRAB = "/tmp/phoxi_grab_build/phoxi_grab"
REMOTE_DEPTH = "/tmp/phoxi_scan_depth.raw"
REMOTE_POINTCLOUD = "/tmp/phoxi_scan_pointcloud.raw"
REMOTE_TEXTURE = "/tmp/phoxi_scan_texture.raw"

# Raw file format: 8-byte header (width, height as int32), then data
HEADER_SIZE = 8


def _parse_raw(data: bytes, dtype: np.dtype, channels: int = 1) -> np.ndarray:
    """Parse a phoxi_grab raw file (8-byte header + flat data)."""
    w, h = struct.unpack("ii", data[:HEADER_SIZE])
    arr = np.frombuffer(data[HEADER_SIZE:], dtype=dtype)
    if channels > 1:
        return arr.reshape(h, w, channels)
    return arr.reshape(h, w)


class PhoxiClient:
    """SSH wrapper for PhotoNeo phoxi_grab on a remote host."""

    def __init__(self, host: str = "tuppy", user: str = "doug"):
        self.host = host
        self.user = user
        self._connected = False
        self._resolution = None
        self._depth_transformer = None
        # Cached scan data
        self.depth: np.ndarray | None = None
        self.pointcloud: np.ndarray | None = None
        self.texture: np.ndarray | None = None

    async def _ssh(self, cmd: str, timeout: float = 30.0) -> tuple[str, str, int]:
        """Run a command on the remote host via SSH."""
        ssh_cmd = [
            "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
            f"{self.user}@{self.host}", cmd,
        ]
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"SSH command timed out after {timeout}s: {cmd}")
        return stdout.decode(), stderr.decode(), proc.returncode

    async def _scp_from(self, remote_path: str, local_path: str, timeout: float = 30.0):
        """Copy a file from the remote host."""
        scp_cmd = [
            "scp", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
            f"{self.user}@{self.host}:{remote_path}", local_path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *scp_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"SCP timed out after {timeout}s")
        if proc.returncode != 0:
            raise RuntimeError(f"SCP failed: {stderr.decode()}")

    async def connect(self) -> dict:
        """Verify connectivity to tuppy and phoxi_grab availability."""
        try:
            stdout, stderr, rc = await self._ssh(
                f"hostname && test -x {PHOXI_GRAB} && echo OK || echo MISSING"
            )
            lines = stdout.strip().split("\n")
            hostname = lines[0] if lines else "unknown"
            available = "OK" in stdout

            if not available:
                return {
                    "connected": False,
                    "error": f"phoxi_grab not found at {PHOXI_GRAB} on {hostname}",
                }

            # Check PhoXi Control is running
            stdout2, _, _ = await self._ssh("pgrep -c PhoXiControl")
            phoxi_running = stdout2.strip() != "0"

            self._connected = True
            return {
                "connected": True,
                "host": self.host,
                "hostname": hostname,
                "phoxi_grab": PHOXI_GRAB,
                "phoxi_control_running": phoxi_running,
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    async def capture(self) -> dict:
        """Trigger a PhotoNeo scan and retrieve depth + pointcloud + texture."""
        if not self._connected:
            return {"error": "Not connected. Call connect first."}

        # Run phoxi_grab on tuppy
        stdout, stderr, rc = await self._ssh(PHOXI_GRAB, timeout=30.0)
        if rc != 0:
            return {"error": f"phoxi_grab failed (rc={rc}): {stderr}"}

        # Parse output for resolution
        resolution = None
        for line in stdout.split("\n"):
            if "Resolution:" in line:
                parts = line.split(":")[-1].strip().split("x")
                if len(parts) == 2:
                    resolution = (int(parts[0]), int(parts[1]))

        # SCP the raw files back
        with tempfile.TemporaryDirectory() as tmpdir:
            local_files = {}
            for name, remote in [
                ("depth", REMOTE_DEPTH),
                ("pointcloud", REMOTE_POINTCLOUD),
                ("texture", REMOTE_TEXTURE),
            ]:
                local_path = f"{tmpdir}/{name}.raw"
                await self._scp_from(remote, local_path)
                local_files[name] = Path(local_path).read_bytes()

            # Parse raw data
            self.depth = _parse_raw(local_files["depth"], np.float32)
            self.pointcloud = _parse_raw(local_files["pointcloud"], np.float32, channels=3)
            self.texture = _parse_raw(local_files["texture"], np.uint16)

        self._resolution = (self.depth.shape[1], self.depth.shape[0])

        # Compute stats on valid pixels
        valid_mask = self.depth > 0
        valid_count = int(valid_mask.sum())
        total = self.depth.size

        stats = {
            "width": self._resolution[0],
            "height": self._resolution[1],
            "valid_pixels": valid_count,
            "total_pixels": total,
            "coverage": round(valid_count / total, 3),
        }

        if valid_count > 0:
            stats["depth_range_mm"] = {
                "min": round(float(self.depth[valid_mask].min()), 1),
                "max": round(float(self.depth[valid_mask].max()), 1),
                "mean": round(float(self.depth[valid_mask].mean()), 1),
            }
            pc_valid = self.pointcloud[valid_mask]
            stats["pointcloud_bounds_mm"] = {
                "x": [round(float(pc_valid[:, 0].min()), 1), round(float(pc_valid[:, 0].max()), 1)],
                "y": [round(float(pc_valid[:, 1].min()), 1), round(float(pc_valid[:, 1].max()), 1)],
                "z": [round(float(pc_valid[:, 2].min()), 1), round(float(pc_valid[:, 2].max()), 1)],
            }

        return {"success": True, **stats}

    def get_depth_at(self, pixel_x: int, pixel_y: int) -> dict:
        """Get depth and 3D position at a specific pixel coordinate."""
        if self.depth is None:
            return {"error": "No scan data. Call capture_depth first."}

        h, w = self.depth.shape
        if not (0 <= pixel_x < w and 0 <= pixel_y < h):
            return {"error": f"Pixel ({pixel_x}, {pixel_y}) out of bounds ({w}x{h})"}

        depth_mm = float(self.depth[pixel_y, pixel_x])
        xyz = self.pointcloud[pixel_y, pixel_x]

        if depth_mm == 0:
            return {
                "pixel": [pixel_x, pixel_y],
                "valid": False,
                "depth_mm": 0,
                "note": "No depth data at this pixel (shadow/occlusion)",
            }

        return {
            "pixel": [pixel_x, pixel_y],
            "valid": True,
            "depth_mm": round(depth_mm, 1),
            "position_mm": {
                "x": round(float(xyz[0]), 1),
                "y": round(float(xyz[1]), 1),
                "z": round(float(xyz[2]), 1),
            },
        }

    def get_depth_patch(self, pixel_x: int, pixel_y: int, radius: int = 5) -> dict:
        """Get median depth/position in a patch around a pixel (more robust)."""
        if self.depth is None:
            return {"error": "No scan data. Call capture_depth first."}

        h, w = self.depth.shape
        y0 = max(0, pixel_y - radius)
        y1 = min(h, pixel_y + radius + 1)
        x0 = max(0, pixel_x - radius)
        x1 = min(w, pixel_x + radius + 1)

        patch_depth = self.depth[y0:y1, x0:x1]
        patch_pc = self.pointcloud[y0:y1, x0:x1]
        valid = patch_depth > 0

        if valid.sum() == 0:
            return {
                "pixel": [pixel_x, pixel_y],
                "radius": radius,
                "valid": False,
                "note": "No valid depth in patch",
            }

        median_depth = float(np.median(patch_depth[valid]))
        median_xyz = np.median(patch_pc[valid], axis=0)

        return {
            "pixel": [pixel_x, pixel_y],
            "radius": radius,
            "valid": True,
            "valid_pixels": int(valid.sum()),
            "depth_mm": round(median_depth, 1),
            "position_mm": {
                "x": round(float(median_xyz[0]), 1),
                "y": round(float(median_xyz[1]), 1),
                "z": round(float(median_xyz[2]), 1),
            },
        }

    def camera_to_robot(self, x_mm: float, y_mm: float, z_mm: float) -> dict:
        """Transform a 3D point from camera frame (mm) to robot frame (m).

        Loads calibration on first call. Returns error if not calibrated.
        """
        if self._depth_transformer is None:
            from common.depth_calibration import get_depth_transformer
            self._depth_transformer = get_depth_transformer()

        if self._depth_transformer is None:
            return {"error": "No depth calibration. Run scripts/calibrate_depth.py first."}

        rx, ry, rz = self._depth_transformer.camera_to_robot(x_mm, y_mm, z_mm)
        return {
            "robot_coords_m": {
                "x": round(rx, 4),
                "y": round(ry, 4),
                "z": round(rz, 4),
            }
        }

    def save_npz(self, path: str = "/tmp/phoxi_scan.npz") -> dict:
        """Save current scan data as NPZ for external use."""
        if self.depth is None:
            return {"error": "No scan data. Call capture_depth first."}

        np.savez_compressed(
            path,
            depth=self.depth,
            pointcloud=self.pointcloud,
            texture=self.texture,
        )
        return {"success": True, "path": path, "size_bytes": Path(path).stat().st_size}


# Singleton
_client: PhoxiClient | None = None


def get_phoxi_client() -> PhoxiClient:
    global _client
    if _client is None:
        _client = PhoxiClient()
    return _client
