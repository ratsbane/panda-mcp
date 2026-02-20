#!/usr/bin/env python3
"""
DINOv2 embedding server. Runs on Spark (GPU), accepts images via HTTP,
returns embedding vectors.

Usage:
    python3 embedding_server.py [--port 8091] [--model dinov2_vits14]

The server accepts POST /embed with a JPEG image body and returns
a JSON response with the embedding vector.
"""

import argparse
import io
import json
import logging
import time

import os
# Limit PyTorch CPU threads BEFORE importing torch — prevents spawning
# dozens of OpenMP/MKL threads on machines with many cores (e.g. DGX Spark).
# We only need GPU inference, so 2 CPU threads is plenty.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
torch.set_num_threads(4)

from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger(__name__)

# Global model reference
model = None
transform = None
device = None
embed_dim = None


def load_model(model_name: str = "dinov2_vits14"):
    """Load DINOv2 model onto GPU."""
    global model, transform, device, embed_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading {model_name} on {device}...")

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()
    embed_dim = model.embed_dim

    # DINOv2 expects 224x224 images, normalized with ImageNet stats
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    logger.info(f"Model loaded: embed_dim={embed_dim}")


def embed_image(image_bytes: bytes) -> np.ndarray:
    """Compute DINOv2 embedding for a JPEG image."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding.cpu().numpy().flatten()


def embed_crop(image_bytes: bytes, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Compute DINOv2 embedding for a crop of an image."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    crop = img.crop((x1, y1, x2, y2))
    tensor = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding.cpu().numpy().flatten()


class EmbeddingHandler(BaseHTTPRequestHandler):
    """HTTP handler for embedding requests."""

    def do_POST(self):
        if self.path == "/embed":
            content_length = int(self.headers["Content-Length"])
            image_bytes = self.rfile.read(content_length)

            t0 = time.time()
            embedding = embed_image(image_bytes)
            dt = time.time() - t0

            response = {
                "embedding": embedding.tolist(),
                "dim": len(embedding),
                "time_ms": round(dt * 1000, 1),
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        elif self.path == "/embed_crop":
            # Expects multipart: JSON metadata + image
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)

            # Parse JSON header (first line) + image data
            # Format: JSON line \n image_bytes
            newline_idx = body.index(b"\n")
            metadata = json.loads(body[:newline_idx])
            image_bytes = body[newline_idx + 1:]

            t0 = time.time()
            embedding = embed_crop(
                image_bytes,
                metadata["x1"], metadata["y1"],
                metadata["x2"], metadata["y2"],
            )
            dt = time.time() - t0

            response = {
                "embedding": embedding.tolist(),
                "dim": len(embedding),
                "time_ms": round(dt * 1000, 1),
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            response = {
                "model": "dinov2_vits14",
                "embed_dim": embed_dim,
                "device": str(device),
                "ready": model is not None,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        logger.debug(format % args)


def check_port_available(port: int) -> bool:
    """Check if port is available. If not, another instance is already running."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="DINOv2 embedding server")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14",
                                 "dinov2_vitl14", "dinov2_vitg14"])
    args = parser.parse_args()

    # Prevent multiple instances — check before loading model (which uses ~1.2GB RAM)
    if not check_port_available(args.port):
        logger.error(f"Port {args.port} already in use — another instance is running. Exiting.")
        return

    load_model(args.model)

    server = HTTPServer(("0.0.0.0", args.port), EmbeddingHandler)
    logger.info(f"Embedding server listening on port {args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
