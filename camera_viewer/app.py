"""Camera viewer web app - live 2D USB camera stream + depth camera images + 3D fusion."""

import asyncio
import logging
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import zmq
import zmq.asyncio
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

ZMQ_IPC = "ipc:///tmp/camera-daemon.sock"
ZMQ_TCP = "tcp://127.0.0.1:5555"

LOCAL_DEPTH = "/tmp/phoxi_scan_depth.raw"
LOCAL_TEXTURE = "/tmp/phoxi_scan_texture.raw"
LOCAL_POINTCLOUD = "/tmp/phoxi_scan_pointcloud.raw"

REMOTE_HOST = "tuppy"
PHOXI_GRAB = "/tmp/phoxi_grab_build/phoxi_grab"
REMOTE_DEPTH = "/tmp/phoxi_scan_depth.raw"
REMOTE_TEXTURE = "/tmp/phoxi_scan_texture.raw"
REMOTE_POINTCLOUD = "/tmp/phoxi_scan_pointcloud.raw"

HEADER_SIZE = 8


def _parse_raw(data: bytes, dtype: np.dtype, channels: int = 1) -> np.ndarray:
    """Parse a phoxi_grab raw file (8-byte header + flat data)."""
    w, h = struct.unpack("ii", data[:HEADER_SIZE])
    arr = np.frombuffer(data[HEADER_SIZE:], dtype=dtype)
    if channels > 1:
        return arr.reshape(h, w, channels)
    return arr.reshape(h, w)


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Camera Viewer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a1a; color: #eee; font-family: system-ui, sans-serif; padding: 16px; }
  h1 { font-size: 1.2rem; margin-bottom: 12px; color: #aaa; }
  h2 { font-size: 0.95rem; margin-bottom: 6px; color: #888; font-weight: normal; }
  .container { display: flex; flex-direction: column; gap: 12px; }
  .depth-row { display: flex; gap: 12px; }
  .depth-row > .panel { flex: 1; min-width: 0; }
  .panel { background: #222; border-radius: 6px; padding: 8px; display: flex;
           flex-direction: column; }
  .panel img { width: 100%; height: auto; object-fit: contain; border-radius: 4px; background: #111; }
  .controls { margin-top: 8px; display: flex; gap: 8px; align-items: center; }
  button { background: #335; color: #ccc; border: 1px solid #446; border-radius: 4px;
           padding: 6px 14px; cursor: pointer; font-size: 0.85rem; }
  button:hover { background: #447; }
  button:disabled { opacity: 0.5; cursor: wait; }
  .status { font-size: 0.8rem; color: #666; }
  .placeholder { color: #555; font-size: 0.85rem; padding: 20px; text-align: center;
                 flex: 1; display: flex; align-items: center; justify-content: center; }
</style>
</head>
<body>
<h1>Camera Viewer</h1>
<div class="container">
  <div class="panel">
    <h2>USB Camera (live)</h2>
    <img id="stream" alt="2D camera stream" style="display:none">
    <div id="stream-placeholder" class="placeholder">Connecting to camera...</div>
  </div>
  <div class="panel">
    <h2>3D Fusion (YOLO + Depth)</h2>
    <img id="fusion" alt="3D fusion overlay" style="display:none">
    <div id="fusion-placeholder" class="placeholder">Click "Capture New Scan" to generate</div>
    <div class="controls">
      <button id="fusion-btn" onclick="refreshFusion()">Refresh Fusion</button>
      <span id="fusion-status" class="status"></span>
    </div>
  </div>
  <div class="depth-row">
    <div class="panel">
      <h2>Depth Texture</h2>
      <img id="texture" alt="Depth texture" style="display:none">
      <div id="texture-placeholder" class="placeholder">No scan yet</div>
    </div>
    <div class="panel">
      <h2>Depth Colormap</h2>
      <img id="colormap" alt="Depth colormap" style="display:none">
      <div id="colormap-placeholder" class="placeholder">No scan yet</div>
    </div>
  </div>
  <div class="controls">
    <button id="capture-btn" onclick="captureDepth()">Capture New Scan</button>
    <span id="depth-status" class="status"></span>
  </div>
</div>
<script>
  // Start stream after page loads (don't block rendering)
  window.addEventListener('load', function() {
    var img = document.getElementById('stream');
    img.onload = function() {
      img.style.display = '';
      document.getElementById('stream-placeholder').style.display = 'none';
    };
    img.onerror = function() {
      document.getElementById('stream-placeholder').textContent = 'Camera stream unavailable';
    };
    img.src = '/stream';
  });

  // Load depth image with placeholder handling
  function loadImg(id, url) {
    var img = document.getElementById(id);
    var ph = document.getElementById(id + '-placeholder');
    var tmp = new Image();
    tmp.onload = function() {
      img.src = url;
      img.style.display = '';
      if (ph) ph.style.display = 'none';
    };
    tmp.onerror = function() {
      // keep placeholder visible
    };
    tmp.src = url;
  }

  function refreshDepth() {
    var t = Date.now();
    loadImg('texture', '/depth/texture?t=' + t);
    loadImg('colormap', '/depth/colormap?t=' + t);
  }

  function refreshFusion() {
    var btn = document.getElementById('fusion-btn');
    var status = document.getElementById('fusion-status');
    btn.disabled = true;
    status.textContent = 'Running YOLO + depth fusion...';
    var t = Date.now();
    var img = document.getElementById('fusion');
    var ph = document.getElementById('fusion-placeholder');
    var tmp = new Image();
    tmp.onload = function() {
      img.src = tmp.src;
      img.style.display = '';
      if (ph) ph.style.display = 'none';
      btn.disabled = false;
      status.textContent = '';
    };
    tmp.onerror = function() {
      btn.disabled = false;
      status.textContent = 'Fusion failed (no depth scan?)';
    };
    tmp.src = '/fusion?t=' + t;
  }

  // Initial load + auto-refresh
  refreshDepth();
  setInterval(refreshDepth, 5000);

  // Capture button
  async function captureDepth() {
    var btn = document.getElementById('capture-btn');
    var status = document.getElementById('depth-status');
    btn.disabled = true;
    status.textContent = 'Scanning...';
    try {
      var resp = await fetch('/depth/capture');
      var data = await resp.json();
      if (data.error) {
        status.textContent = 'Error: ' + data.error;
      } else {
        status.textContent = 'Scan complete (' + (data.depth_shape || '?') + ')';
        refreshDepth();
        refreshFusion();
      }
    } catch (e) {
      status.textContent = 'Failed: ' + e.message;
    }
    btn.disabled = false;
  }
</script>
</body>
</html>"""


async def homepage(request):
    return HTMLResponse(HTML_PAGE)


async def mjpeg_stream(request):
    """MJPEG stream from the USB camera daemon via ZeroMQ."""

    async def generate():
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.setsockopt_string(zmq.SUBSCRIBE, "frame")

        connected = False
        for endpoint in [ZMQ_IPC, ZMQ_TCP]:
            try:
                sock.connect(endpoint)
                # Test receive
                parts = await asyncio.wait_for(
                    sock.recv_multipart(), timeout=3.0
                )
                connected = True
                logger.info(f"Connected to camera at {endpoint}")
                break
            except Exception:
                sock.disconnect(endpoint)

        if not connected:
            sock.close()
            ctx.term()
            return

        try:
            # Yield the first frame we already received
            if len(parts) == 3:
                jpeg_bytes = parts[2]
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )

            while True:
                try:
                    parts = await asyncio.wait_for(
                        sock.recv_multipart(), timeout=5.0
                    )
                    if len(parts) == 3:
                        jpeg_bytes = parts[2]
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + jpeg_bytes
                            + b"\r\n"
                        )
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
        finally:
            sock.close()
            ctx.term()

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _load_texture_jpeg() -> bytes | None:
    """Load depth texture raw file and return as JPEG bytes."""
    path = Path(LOCAL_TEXTURE)
    if not path.exists():
        return None
    data = path.read_bytes()
    if len(data) <= HEADER_SIZE:
        return None
    texture = _parse_raw(data, np.uint16)
    # PhotoNeo texture is 10-bit (0-1023) in uint16
    normalized = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, jpeg = cv2.imencode(".jpg", normalized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpeg.tobytes()


def _load_depth_colormap_jpeg() -> bytes | None:
    """Load depth raw file and return as colormapped JPEG bytes."""
    path = Path(LOCAL_DEPTH)
    if not path.exists():
        return None
    data = path.read_bytes()
    if len(data) <= HEADER_SIZE:
        return None
    depth = _parse_raw(data, np.float32)
    # Mask invalid (0) depth
    valid = depth > 0
    if not valid.any():
        return None
    # Normalize valid depths to 0-255
    d_min, d_max = depth[valid].min(), depth[valid].max()
    normalized = np.zeros_like(depth, dtype=np.uint8)
    if d_max > d_min:
        normalized[valid] = (
            (depth[valid] - d_min) / (d_max - d_min) * 255
        ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    # Black out invalid pixels
    colored[~valid] = 0
    _, jpeg = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpeg.tobytes()


async def depth_texture(request):
    jpeg = _load_texture_jpeg()
    if jpeg is None:
        return Response("No depth texture available", status_code=404)
    return Response(jpeg, media_type="image/jpeg")


async def depth_colormap(request):
    jpeg = _load_depth_colormap_jpeg()
    if jpeg is None:
        return Response("No depth data available", status_code=404)
    return Response(jpeg, media_type="image/jpeg")


async def depth_capture(request):
    """Trigger a new depth scan via SSH to tuppy."""
    try:
        # Run phoxi_grab
        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
            REMOTE_HOST, PHOXI_GRAB,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if proc.returncode != 0:
            return JSONResponse({"error": f"phoxi_grab failed: {stderr.decode()}"})

        # SCP files back
        for remote, local in [
            (REMOTE_DEPTH, LOCAL_DEPTH),
            (REMOTE_TEXTURE, LOCAL_TEXTURE),
            (REMOTE_POINTCLOUD, LOCAL_POINTCLOUD),
        ]:
            proc = await asyncio.create_subprocess_exec(
                "scp", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                f"{REMOTE_HOST}:{remote}", local,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15.0)
            if proc.returncode != 0:
                return JSONResponse({"error": f"SCP failed for {remote}"})

        # Parse to get shape info
        depth_data = Path(LOCAL_DEPTH).read_bytes()
        depth = _parse_raw(depth_data, np.float32)
        return JSONResponse({
            "status": "ok",
            "depth_shape": f"{depth.shape[1]}x{depth.shape[0]}",
        })

    except asyncio.TimeoutError:
        return JSONResponse({"error": "Scan timed out"})
    except Exception as e:
        return JSONResponse({"error": str(e)})


def _grab_usb_frame() -> np.ndarray | None:
    """Grab a single BGR frame from the camera daemon via ZeroMQ (sync)."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 3000)
    sock.setsockopt_string(zmq.SUBSCRIBE, "frame")
    try:
        for endpoint in [ZMQ_IPC, ZMQ_TCP]:
            try:
                sock.connect(endpoint)
                parts = sock.recv_multipart()
                if len(parts) == 3:
                    jpeg_bytes = parts[2]
                    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                try:
                    sock.disconnect(endpoint)
                except Exception:
                    pass
        return None
    finally:
        sock.close()
        ctx.term()


# Source colors for annotation: fused=green, depth_only=cyan, yolo_only=yellow
_SOURCE_COLORS = {
    "fused": (0, 220, 0),
    "depth_only": (220, 180, 0),
    "yolo_only": (0, 220, 220),
}


def _render_fusion_image(
    frame: np.ndarray,
    scene_dict: dict,
    usb_calibration_path: str = "/home/doug/panda-mcp/calibration/aruco_calibration.npz",
) -> np.ndarray:
    """Draw 3D fusion results on a USB camera frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Load inverse homography (robot XY -> pixel) for projecting 3D positions
    cal_path = Path(usb_calibration_path)
    H_inv = None
    if cal_path.exists():
        cal = np.load(str(cal_path), allow_pickle=True)
        H = cal["H"]
        H_inv = np.linalg.inv(H)

    objects = scene_dict.get("objects", [])

    for obj in objects:
        source = obj.get("source", "fused")
        color = _SOURCE_COLORS.get(source, (0, 220, 0))
        label = obj.get("label", "?")
        conf = obj.get("confidence", 0)
        pos = obj.get("position_m", {})
        dims = obj.get("dimensions_m", {})
        grasp_w = obj.get("grasp_width_m")
        point_count = obj.get("point_count", 0)

        rx, ry, rz = pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)

        # Project robot XY to pixel using inverse homography
        px, py = None, None
        if H_inv is not None:
            pt = np.array([[rx, ry]], dtype=np.float32).reshape(-1, 1, 2)
            pixel = cv2.perspectiveTransform(pt, H_inv)
            px, py = int(pixel[0, 0, 0]), int(pixel[0, 0, 1])

        if px is None or not (0 <= px < w and 0 <= py < h):
            # Off-screen — skip drawing but could add to a legend
            continue

        # Draw crosshair
        size = 18
        cv2.line(annotated, (px - size, py), (px + size, py), color, 2)
        cv2.line(annotated, (px, py - size), (px, py + size), color, 2)
        cv2.circle(annotated, (px, py), 4, color, -1)

        # Build label text
        if source == "fused" or source == "yolo_only":
            text1 = f"{label} {conf:.0%}"
        else:
            text1 = f"unknown ({point_count}pts)"

        text2 = f"({rx:.2f}, {ry:.2f}, {rz:.3f})"

        # Draw text background + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw1, th1), _ = cv2.getTextSize(text1, font, scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(text2, font, scale * 0.8, thickness)
        tw = max(tw1, tw2)
        total_h = th1 + th2 + 12

        # Position label above crosshair
        lx = max(0, min(px - tw // 2, w - tw - 4))
        ly = max(total_h + 4, py - size - 6)

        cv2.rectangle(annotated, (lx - 2, ly - th1 - 4), (lx + tw + 4, ly + th2 + 8),
                      (0, 0, 0), -1)
        cv2.rectangle(annotated, (lx - 2, ly - th1 - 4), (lx + tw + 4, ly + th2 + 8),
                      color, 1)
        cv2.putText(annotated, text1, (lx + 1, ly), font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(annotated, text2, (lx + 1, ly + th2 + 6), font, scale * 0.8,
                    (180, 180, 180), thickness, cv2.LINE_AA)

        # Draw grasp width indicator if available
        if grasp_w and grasp_w > 0 and H_inv is not None:
            half = grasp_w / 2
            pt_l = np.array([[rx, ry - half]], dtype=np.float32).reshape(-1, 1, 2)
            pt_r = np.array([[rx, ry + half]], dtype=np.float32).reshape(-1, 1, 2)
            px_l = cv2.perspectiveTransform(pt_l, H_inv)
            px_r = cv2.perspectiveTransform(pt_r, H_inv)
            x1, y1 = int(px_l[0, 0, 0]), int(px_l[0, 0, 1])
            x2, y2 = int(px_r[0, 0, 0]), int(px_r[0, 0, 1])
            cv2.line(annotated, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Draw legend
    ly = h - 14
    for src, clr in _SOURCE_COLORS.items():
        cv2.circle(annotated, (10, ly), 5, clr, -1)
        cv2.putText(annotated, src, (20, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    clr, 1, cv2.LINE_AA)
        ly -= 18

    # Summary bar at top
    summary = scene_dict.get("summary", "")
    total = scene_dict.get("total_objects", 0)
    fused = scene_dict.get("fused_count", 0)
    header = f"Objects: {total} (fused: {fused})"
    cv2.rectangle(annotated, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(annotated, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


async def fusion_image(request):
    """Render fusion overlay: USB camera frame + YOLO labels + depth 3D positions."""
    try:
        # Grab USB camera frame (sync ZMQ in thread to avoid blocking event loop)
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(None, _grab_usb_frame)
        if frame is None:
            return Response("Camera unavailable", status_code=503)

        # Run fusion (CPU-bound with Hailo inference, run in executor)
        def _run_fusion():
            from common.depth_fusion import build_scene_graph
            scene = build_scene_graph(
                frame,
                depth_npz_path="/tmp/phoxi_scan.npz",
                max_age_seconds=300,
                confidence_threshold=0.3,
            )
            return scene.to_dict()

        scene_dict = await loop.run_in_executor(None, _run_fusion)

        # Render annotated image
        annotated = _render_fusion_image(frame, scene_dict)
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return Response(jpeg.tobytes(), media_type="image/jpeg")

    except Exception as e:
        logger.exception("Fusion image failed")
        return Response(f"Fusion failed: {e}", status_code=500)


JOG_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>Remote Arm Control</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; touch-action: none; }
  html, body { height: 100%; overflow: auto; }
  body { background: #111; color: #eee; font-family: system-ui, -apple-system, sans-serif; }

  .app { display: flex; flex-direction: column; height: 100vh; }

  /* Top bar */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 12px; background: #1a1a1a; border-bottom: 1px solid #333;
    flex-shrink: 0; z-index: 10;
  }
  .topbar h1 { font-size: 0.9rem; color: #aaa; font-weight: 500; }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%; display: inline-block;
    margin-right: 6px; vertical-align: middle;
  }
  .status-dot.connected { background: #4c4; }
  .status-dot.disconnected { background: #c44; }
  .status-text { font-size: 0.75rem; color: #888; }

  /* Main content row: joint sidebar + camera */
  .content-row {
    flex: 1; display: flex; min-height: 0; overflow: hidden;
  }

  /* Camera area */
  .camera-area {
    flex: 1; position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
    background: #000; min-height: 0;
  }
  .camera-area img {
    max-width: 100%; max-height: 100%; object-fit: contain;
  }
  .camera-placeholder {
    color: #444; font-size: 0.9rem;
  }

  /* Position overlay on camera */
  .pos-overlay {
    position: absolute; top: 8px; left: 8px;
    background: rgba(0,0,0,0.7); padding: 4px 10px; border-radius: 4px;
    font-size: 0.75rem; font-family: 'SF Mono', 'Consolas', monospace;
    color: #aaa; pointer-events: none;
  }

  /* Controls area */
  .controls-area {
    flex-shrink: 0; background: #1a1a1a; border-top: 1px solid #333;
    padding: 8px 12px;
  }

  /* Button row */
  .btn-row {
    display: flex; gap: 8px; justify-content: center; margin-bottom: 8px;
    flex-wrap: wrap;
  }
  .btn {
    padding: 10px 16px; border: 1px solid #444; border-radius: 6px;
    background: #252525; color: #ddd; font-size: 0.8rem; font-weight: 500;
    cursor: pointer; min-width: 64px; text-align: center;
    transition: background 0.1s;
    -webkit-user-select: none; user-select: none;
  }
  .btn:active, .btn.active { background: #446; border-color: #668; }
  .btn.grasp { border-color: #4a4; }
  .btn.grasp:active, .btn.grasp.active { background: #363; }
  .btn.open { border-color: #48a; }
  .btn.open:active, .btn.open.active { background: #346; }
  .btn.danger { border-color: #a44; }
  .btn.danger:active { background: #633; }
  .speed-label { font-size: 0.75rem; color: #888; align-self: center; min-width: 80px; text-align: center; }

  /* Joystick area */
  .joy-row {
    display: flex; gap: 12px; justify-content: center; align-items: center;
    padding: 4px 0;
  }
  .joy-container {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
  }
  .joy-label { font-size: 0.65rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }

  .joystick {
    position: relative; width: 120px; height: 120px;
    border-radius: 50%; background: #222; border: 2px solid #444;
    cursor: grab;
  }
  .joystick .knob {
    position: absolute; width: 44px; height: 44px;
    border-radius: 50%; background: #556; border: 2px solid #778;
    top: 50%; left: 50%; transform: translate(-50%, -50%);
    pointer-events: none; transition: background 0.1s;
  }
  .joystick.active .knob { background: #668; border-color: #99b; }

  /* Z slider (vertical only) */
  .joystick.z-only { width: 60px; }

  /* D-pad */
  .dpad {
    display: grid; grid-template-columns: 36px 36px 36px; grid-template-rows: 36px 36px 36px;
    gap: 2px;
  }
  .dpad .btn { padding: 0; min-width: 36px; height: 36px; font-size: 0.75rem;
               display: flex; align-items: center; justify-content: center; }
  .dpad .center { background: transparent; border: none; cursor: default; }

  /* Input source indicator */
  .input-source {
    font-size: 0.65rem; color: #555; text-align: center; margin-top: 4px;
  }

  /* Joint sidebar */
  .joint-panel {
    flex-shrink: 0; width: 140px; background: #151515; border-right: 1px solid #333;
    padding: 6px 8px; display: flex; flex-direction: column;
    font-family: 'SF Mono', 'Consolas', monospace;
  }
  .joint-panel-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 4px;
  }
  .joint-panel-header h2 { font-size: 0.65rem; color: #555; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
  .joint-panel-header .toggle { display: none; }
  .joint-rows { flex: 1; display: flex; flex-direction: column; gap: 1px; }
  .joint-row {
    display: flex; align-items: center; gap: 4px;
    font-size: 0.6rem;
  }
  .joint-label { width: 16px; color: #666; text-align: right; flex-shrink: 0; }
  .joint-bar-container {
    flex: 1; height: 8px; background: #222; border-radius: 2px;
    position: relative; overflow: hidden;
  }
  .joint-bar-fill {
    position: absolute; top: 0; bottom: 0; background: #335; border-radius: 2px;
    transition: left 0.15s, width 0.15s;
  }
  .joint-bar-marker {
    position: absolute; top: -1px; bottom: -1px; width: 2px;
    background: #8af; border-radius: 1px;
    transition: left 0.15s;
  }
  .joint-bar-center {
    position: absolute; top: 2px; bottom: 2px; width: 1px;
    background: #444; left: 50%;
  }
  .joint-bar-container.warn .joint-bar-marker { background: #fa4; }
  .joint-bar-container.danger .joint-bar-marker { background: #f44; }
  .joint-value { width: 36px; color: #666; text-align: right; flex-shrink: 0; font-size: 0.55rem; }
  .joint-limits { display: none; }
  .orient-row {
    margin-top: auto; padding-top: 4px; border-top: 1px solid #282828;
    font-size: 0.55rem; color: #555; line-height: 1.4;
  }

  /* Responsive: smaller screens */
  @media (max-height: 500px) {
    .joystick { width: 90px; height: 90px; }
    .joystick .knob { width: 34px; height: 34px; }
    .joystick.z-only { width: 50px; }
    .btn { padding: 8px 12px; font-size: 0.75rem; }
  }
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <h1>Remote Arm Control</h1>
    <span class="status-text">
      <span id="ws-dot" class="status-dot disconnected"></span>
      <span id="ws-status">Connecting...</span>
    </span>
  </div>

  <div class="content-row">
    <div class="joint-panel" id="joint-panel">
      <div class="joint-panel-header"><h2>Joints</h2></div>
      <div class="joint-rows" id="joint-rows"></div>
      <div class="orient-row" id="orient-row"></div>
    </div>
    <div class="camera-area">
      <img id="stream" alt="Camera" style="display:none">
      <div id="stream-ph" class="camera-placeholder">Connecting to camera...</div>
      <div id="pos-overlay" class="pos-overlay" style="display:none">
        <div id="pos-text">X: --- Y: --- Z: ---</div>
        <div id="grip-text">Gripper: ---</div>
        <div id="ik-warn" style="display:none; color:#f64; margin-top:2px; font-weight:600;">IK BLOCKED</div>
      </div>
    </div>
  </div>

  <div class="controls-area">
    <div class="btn-row">
      <button class="btn grasp" id="btn-grasp" onpointerdown="sendEvent('grasp')">Grasp</button>
      <button class="btn open" id="btn-open" onpointerdown="sendEvent('open_gripper')">Open</button>
      <button class="btn" id="btn-home" onpointerdown="sendEvent('home')">Home</button>
      <button class="btn" id="btn-speed" onpointerdown="cycleSpeed()">Speed</button>
      <span class="speed-label" id="speed-label">medium</span>
      <button class="btn danger" id="btn-stop" onpointerdown="sendEvent('stop_jog')">STOP</button>
    </div>
    <div class="joy-row">
      <div class="joy-container">
        <div class="joy-label">Move XY</div>
        <div class="joystick" id="joy-xy">
          <div class="knob" id="knob-xy"></div>
        </div>
      </div>
      <div class="joy-container">
        <div class="joy-label">D-Pad</div>
        <div class="dpad">
          <div></div>
          <button class="btn" onpointerdown="adjustPitch(1)">&#9650;</button>
          <div></div>
          <button class="btn" onpointerdown="adjustYaw(-1)">&#9664;</button>
          <div class="center"></div>
          <button class="btn" onpointerdown="adjustYaw(1)">&#9654;</button>
          <div></div>
          <button class="btn" onpointerdown="adjustPitch(-1)">&#9660;</button>
          <div></div>
        </div>
      </div>
      <div class="joy-container">
        <div class="joy-label">Height Z</div>
        <div class="joystick z-only" id="joy-z">
          <div class="knob" id="knob-z"></div>
        </div>
      </div>
    </div>
    <div class="input-source" id="input-source">Touch or mouse to control</div>
  </div>

</div>

<script>
// --- Configuration ---
const WS_PORT = 8766;
const SEND_HZ = 20;
const DEADZONE = 0.15;
const SPEED_PRESETS = {slow: 0.015, medium: 0.040, fast: 0.070};
const SPEED_NAMES = ['slow', 'medium', 'fast'];
const ANGLE_STEP = 0.05;

// --- State ---
let ws = null;
let wsConnected = false;
let speedIdx = 1; // medium
let pitch = 0, yaw = 0;
let joyXY = {x: 0, y: 0};
let joyZ = {y: 0};
let sendInterval = null;
let gamepadActive = false;
let gamepadId = null;
let prevButtons = {};  // Edge detection for gamepad buttons

// --- WebSocket ---
function connectWS() {
  const host = window.location.hostname || 'localhost';
  const url = 'ws://' + host + ':' + WS_PORT;
  document.getElementById('ws-status').textContent = 'Connecting...';

  try {
    ws = new WebSocket(url);
  } catch(e) {
    setTimeout(connectWS, 2000);
    return;
  }

  ws.onopen = function() {
    wsConnected = true;
    document.getElementById('ws-dot').className = 'status-dot connected';
    document.getElementById('ws-status').textContent = 'Connected';
  };

  ws.onmessage = function(e) {
    try {
      const data = JSON.parse(e.data);
      if (data.type === 'status') {
        updateStatusDisplay(data);
      }
    } catch(err) {}
  };

  ws.onclose = function() {
    wsConnected = false;
    document.getElementById('ws-dot').className = 'status-dot disconnected';
    document.getElementById('ws-status').textContent = 'Disconnected';
    setTimeout(connectWS, 2000);
  };

  ws.onerror = function() {
    ws.close();
  };
}

function sendState() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const speed = SPEED_NAMES[speedIdx];
  const step = SPEED_PRESETS[speed];

  // Apply deadzone + quadratic response
  let dx = applyResponse(joyXY.y) * step;  // stick up = +X (forward)
  let dy = applyResponse(joyXY.x) * step;  // stick right = +Y
  let dz = applyResponse(joyZ.y) * step;   // stick up = +Z

  // Negate: stick up is negative in browser coords, but we want +X forward
  dx = -dx;
  dz = -dz;

  ws.send(JSON.stringify({
    type: 'state',
    dx: dx, dy: dy, dz: dz,
    pitch: pitch, yaw: yaw,
    speed_name: speed,
    step_size: step,
    fine_mode: false,
    controller: gamepadActive ? ('Gamepad: ' + (gamepadId || '?')) : 'Browser',
  }));
}

function sendEvent(action) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({type: 'event', action: action}));
  // Visual feedback
  const btn = document.getElementById('btn-' + action.replace('_', '-').replace('open-gripper','open'));
  if (btn) { btn.classList.add('active'); setTimeout(() => btn.classList.remove('active'), 200); }
}

function applyResponse(v) {
  if (Math.abs(v) < DEADZONE) return 0;
  const sign = v > 0 ? 1 : -1;
  const norm = (Math.abs(v) - DEADZONE) / (1 - DEADZONE);
  return sign * norm * norm;
}

function cycleSpeed() {
  speedIdx = (speedIdx + 1) % SPEED_NAMES.length;
  document.getElementById('speed-label').textContent = SPEED_NAMES[speedIdx];
}

function adjustPitch(dir) {
  pitch = Math.max(-1, Math.min(1, pitch + dir * ANGLE_STEP));
}

function adjustYaw(dir) {
  yaw = Math.max(-1, Math.min(1, yaw + dir * ANGLE_STEP));
}

const JOINT_NAMES = ['J1','J2','J3','J4','J5','J6','J7'];
let jointLimits = null; // cached from first status message

function initJointBars() {
  const container = document.getElementById('joint-rows');
  container.innerHTML = '';
  for (let i = 0; i < 7; i++) {
    const row = document.createElement('div');
    row.className = 'joint-row';
    row.innerHTML =
      '<span class="joint-label">' + JOINT_NAMES[i] + '</span>' +
      '<span class="joint-limits" id="jlim-' + i + '"></span>' +
      '<div class="joint-bar-container" id="jbar-' + i + '">' +
        '<div class="joint-bar-center"></div>' +
        '<div class="joint-bar-fill" id="jfill-' + i + '"></div>' +
        '<div class="joint-bar-marker" id="jmark-' + i + '"></div>' +
      '</div>' +
      '<span class="joint-value" id="jval-' + i + '"></span>';
    container.appendChild(row);
  }
}

function updateJointDisplay(joints, limits) {
  if (!joints || joints.length < 7) return;
  if (limits && limits.length >= 7 && !jointLimits) jointLimits = limits;
  const lim = jointLimits;
  if (!lim) return;

  for (let i = 0; i < 7; i++) {
    const lo = lim[i][0], hi = lim[i][1];
    const range = hi - lo;
    const pct = ((joints[i] - lo) / range) * 100;
    const margin = 0.10; // same as IK margin
    const nearLimit = (joints[i] < lo + margin) || (joints[i] > hi - margin);
    const veryNear = (joints[i] < lo + 0.05) || (joints[i] > hi - 0.05);

    const bar = document.getElementById('jbar-' + i);
    const marker = document.getElementById('jmark-' + i);
    const fill = document.getElementById('jfill-' + i);
    const val = document.getElementById('jval-' + i);
    const limEl = document.getElementById('jlim-' + i);

    if (marker) marker.style.left = 'calc(' + pct.toFixed(1) + '% - 1.5px)';

    // Fill from center to current position
    const centerPct = ((0 - lo) / range) * 100; // where 0 is on the bar
    if (pct > centerPct) {
      fill.style.left = centerPct + '%';
      fill.style.width = (pct - centerPct) + '%';
    } else {
      fill.style.left = pct + '%';
      fill.style.width = (centerPct - pct) + '%';
    }

    bar.className = 'joint-bar-container' + (veryNear ? ' danger' : nearLimit ? ' warn' : '');
    if (val) val.textContent = joints[i].toFixed(2) + '\u00b0'.replace('\u00b0', '');
    if (val) val.textContent = (joints[i] * 57.296).toFixed(1) + '\u00b0';
    if (limEl) limEl.textContent = (lo * 57.296).toFixed(0) + '\u00b0 .. ' + (hi * 57.296).toFixed(0) + '\u00b0';
  }
}

function updateStatusDisplay(data) {
  const overlay = document.getElementById('pos-overlay');
  overlay.style.display = '';
  const p = data.position || {};
  document.getElementById('pos-text').textContent =
    'X: ' + (p.x || 0).toFixed(3) + '  Y: ' + (p.y || 0).toFixed(3) + '  Z: ' + (p.z || 0).toFixed(3);
  document.getElementById('grip-text').textContent =
    'Gripper: ' + (data.gripper_width || 0).toFixed(3) + 'm';

  // IK blocked indicator
  const ikWarn = document.getElementById('ik-warn');
  ikWarn.style.display = data.ik_blocked ? '' : 'none';

  // Update joint display
  if (data.joints) {
    updateJointDisplay(data.joints, data.joint_limits);
  }

  // Update orientation display
  if (data.orientation) {
    const o = data.orientation;
    document.getElementById('orient-row').innerHTML =
      'R ' + (o.roll * 57.296).toFixed(1) + '\u00b0<br>' +
      'P ' + (o.pitch * 57.296).toFixed(1) + '\u00b0<br>' +
      'Y ' + (o.yaw * 57.296).toFixed(1) + '\u00b0';
  }
}

// --- Virtual Joystick ---
function setupJoystick(containerId, knobId, stateObj, axes) {
  const container = document.getElementById(containerId);
  const knob = document.getElementById(knobId);
  let active = false;
  let rect = null;

  function getPos(clientX, clientY) {
    if (!rect) rect = container.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const maxR = rect.width / 2 - 22; // knob radius
    let dx = (clientX - cx) / maxR;
    let dy = (clientY - cy) / maxR;
    // Clamp to circle
    const dist = Math.sqrt(dx*dx + dy*dy);
    if (dist > 1) { dx /= dist; dy /= dist; }
    return {x: dx, y: dy};
  }

  function updateKnob(nx, ny) {
    const maxPx = (rect ? rect.width : 120) / 2 - 22;
    knob.style.transform = 'translate(calc(-50% + ' + (nx * maxPx) + 'px), calc(-50% + ' + (ny * maxPx) + 'px))';
  }

  function onStart(clientX, clientY) {
    active = true;
    rect = container.getBoundingClientRect();
    container.classList.add('active');
    onMove(clientX, clientY);
  }

  function onMove(clientX, clientY) {
    if (!active) return;
    const pos = getPos(clientX, clientY);
    if (axes === 'xy') { stateObj.x = pos.x; stateObj.y = pos.y; updateKnob(pos.x, pos.y); }
    else if (axes === 'y') { stateObj.y = pos.y; updateKnob(0, pos.y); }
  }

  function onEnd() {
    if (!active) return;
    active = false;
    container.classList.remove('active');
    if (axes === 'xy') { stateObj.x = 0; stateObj.y = 0; }
    else { stateObj.y = 0; }
    knob.style.transform = 'translate(-50%, -50%)';
  }

  // Touch events
  container.addEventListener('touchstart', function(e) {
    e.preventDefault();
    const t = e.touches[0];
    onStart(t.clientX, t.clientY);
  });
  container.addEventListener('touchmove', function(e) {
    e.preventDefault();
    const t = e.touches[0];
    onMove(t.clientX, t.clientY);
  });
  container.addEventListener('touchend', function(e) { e.preventDefault(); onEnd(); });
  container.addEventListener('touchcancel', function(e) { e.preventDefault(); onEnd(); });

  // Mouse events
  container.addEventListener('mousedown', function(e) {
    e.preventDefault();
    onStart(e.clientX, e.clientY);
    function mm(e2) { onMove(e2.clientX, e2.clientY); }
    function mu() { onEnd(); document.removeEventListener('mousemove', mm); document.removeEventListener('mouseup', mu); }
    document.addEventListener('mousemove', mm);
    document.addEventListener('mouseup', mu);
  });
}

// --- Browser Gamepad API ---
function pollGamepad() {
  const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
  let gp = null;
  for (let i = 0; i < gamepads.length; i++) {
    if (gamepads[i] && gamepads[i].connected) { gp = gamepads[i]; break; }
  }

  if (gp) {
    if (!gamepadActive) {
      gamepadActive = true;
      gamepadId = gp.id;
      document.getElementById('input-source').textContent = 'Gamepad: ' + gp.id.substring(0, 40);
    }

    // Map axes (Xbox: 0=LX, 1=LY, 2=RX, 3=RY)
    joyXY.x = gp.axes[0] || 0;
    joyXY.y = gp.axes[1] || 0;
    joyZ.y = gp.axes[3] || 0;  // Right stick Y for Z

    // Update knob visuals
    const xyRect = document.getElementById('joy-xy').getBoundingClientRect();
    const maxXY = xyRect.width / 2 - 22;
    document.getElementById('knob-xy').style.transform =
      'translate(calc(-50% + ' + (joyXY.x * maxXY) + 'px), calc(-50% + ' + (joyXY.y * maxXY) + 'px))';
    const zRect = document.getElementById('joy-z').getBoundingClientRect();
    const maxZ = zRect.width / 2 - 22;
    document.getElementById('knob-z').style.transform =
      'translate(calc(-50% + 0px), calc(-50% + ' + (joyZ.y * maxZ) + 'px))';

    // Buttons (edge-triggered: only fire on press, not while held)
    function btnPressed(idx) {
      const now = gp.buttons[idx] && gp.buttons[idx].pressed;
      const was = prevButtons[idx] || false;
      prevButtons[idx] = now;
      return now && !was;
    }
    if (btnPressed(0)) sendEvent('grasp');
    if (btnPressed(1)) sendEvent('open_gripper');
    if (btnPressed(2)) cycleSpeed();
    if (btnPressed(3)) { pitch = 0; yaw = 0; }
    if (btnPressed(6)) sendEvent('stop_jog');

    // D-pad (edge-triggered)
    if (btnPressed(12)) adjustPitch(1);
    if (btnPressed(13)) adjustPitch(-1);
    if (btnPressed(14)) adjustYaw(-1);
    if (btnPressed(15)) adjustYaw(1);
  } else if (gamepadActive) {
    gamepadActive = false;
    document.getElementById('input-source').textContent = 'Touch or mouse to control';
  }

  requestAnimationFrame(pollGamepad);
}

// --- Init ---
window.addEventListener('load', function() {
  // Camera stream
  var img = document.getElementById('stream');
  img.onload = function() { img.style.display = ''; document.getElementById('stream-ph').style.display = 'none'; };
  img.onerror = function() { document.getElementById('stream-ph').textContent = 'Camera unavailable'; };
  img.src = '/stream';

  // Virtual joysticks
  setupJoystick('joy-xy', 'knob-xy', joyXY, 'xy');
  setupJoystick('joy-z', 'knob-z', joyZ, 'y');

  // Joint display
  initJointBars();

  // WebSocket
  connectWS();

  // Send state at fixed rate
  sendInterval = setInterval(sendState, 1000 / SEND_HZ);

  // Gamepad API polling
  requestAnimationFrame(pollGamepad);
});

// Prevent context menu on long press (mobile)
document.addEventListener('contextmenu', function(e) { e.preventDefault(); });
</script>
</body>
</html>"""


async def jog_page(request):
    return HTMLResponse(JOG_PAGE)


app = Starlette(
    routes=[
        Route("/", homepage),
        Route("/stream", mjpeg_stream),
        Route("/jog", jog_page),
        Route("/fusion", fusion_image),
        Route("/depth/texture", depth_texture),
        Route("/depth/colormap", depth_colormap),
        Route("/depth/capture", depth_capture),
    ],
)
