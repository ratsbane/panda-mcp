"""Camera viewer web app - live 2D USB camera stream + depth camera images."""

import asyncio
import logging
import struct
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


app = Starlette(
    routes=[
        Route("/", homepage),
        Route("/stream", mjpeg_stream),
        Route("/depth/texture", depth_texture),
        Route("/depth/colormap", depth_colormap),
        Route("/depth/capture", depth_capture),
    ],
)
