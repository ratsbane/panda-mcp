#!/usr/bin/env python3
"""
Moondream 2B inference server for object detection and pointing.

Runs on Spark (DGX). Exposes detect() and point() APIs over HTTP.

Usage:
    OMP_NUM_THREADS=4 python moondream_server.py [--port 8091]
"""

import argparse
import base64
import io
import logging
import os
import socket
import sys
import time

# Limit threads before importing torch
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Moondream 2B Server")

# Global model state
model = None
device = None


class DetectRequest(BaseModel):
    image_base64: str
    query: str


class PointRequest(BaseModel):
    image_base64: str
    query: str


class DetectResponse(BaseModel):
    objects: list[dict]
    latency_ms: float


class PointResponse(BaseModel):
    points: list[dict]
    latency_ms: float


def decode_image(b64: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


@app.on_event("startup")
def load_model():
    global model, device
    logger.info("Loading Moondream 2B...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    torch.set_num_threads(4)
    device = "cuda"
    mem = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Model loaded in {time.time()-t0:.1f}s, VRAM: {mem:.1f}GB")


@app.get("/health")
def health():
    return {"status": "ok", "model": "moondream2", "device": str(device)}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    image = decode_image(req.image_base64)
    enc = model.encode_image(image)
    t0 = time.time()
    result = model.detect(enc, req.query)
    latency_ms = (time.time() - t0) * 1000

    objects = result.get("objects", [])
    # Add pixel coordinates and image dimensions
    w, h = image.size
    for obj in objects:
        obj["center_x"] = int((obj["x_min"] + obj["x_max"]) / 2 * w)
        obj["center_y"] = int((obj["y_min"] + obj["y_max"]) / 2 * h)
        obj["bbox_w"] = int((obj["x_max"] - obj["x_min"]) * w)
        obj["bbox_h"] = int((obj["y_max"] - obj["y_min"]) * h)
        obj["image_w"] = w
        obj["image_h"] = h

    return DetectResponse(objects=objects, latency_ms=latency_ms)


@app.post("/point", response_model=PointResponse)
def point(req: PointRequest):
    image = decode_image(req.image_base64)
    enc = model.encode_image(image)
    t0 = time.time()
    result = model.point(enc, req.query)
    latency_ms = (time.time() - t0) * 1000

    points = result.get("points", [])
    # Add pixel coordinates
    w, h = image.size
    for pt in points:
        pt["pixel_x"] = int(pt["x"] * w)
        pt["pixel_y"] = int(pt["y"] * h)
        pt["image_w"] = w
        pt["image_h"] = h

    return PointResponse(points=points, latency_ms=latency_ms)


def check_port(port: int) -> bool:
    """Check if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if not check_port(args.port):
        logger.error(f"Port {args.port} already in use!")
        sys.exit(1)

    logger.info(f"Starting Moondream server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
