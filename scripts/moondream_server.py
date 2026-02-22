#!/usr/bin/env python3
"""
Moondream 2B inference server for object detection, pointing, and skill queries.

Runs on Spark (DGX). Exposes detect(), point(), and query() APIs over HTTP.
Optionally loads LoRA weights for the fine-tuned object-selection model.

Usage:
    OMP_NUM_THREADS=4 python moondream_server.py [--port 8091]
    OMP_NUM_THREADS=4 python moondream_server.py --lora ~/panda-mcp/models/moondream_objsel_r8/best_lora.pt
"""

import argparse
import base64
import io
import logging
import math
import os
import socket
import sys
import time

# Limit threads before importing torch
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
import torch.nn as nn
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
lora_loaded = False
_args = None  # parsed CLI args, set in main()

MD_REVISION = "2025-01-09"


class DetectRequest(BaseModel):
    image_base64: str
    query: str


class PointRequest(BaseModel):
    image_base64: str
    query: str


class QueryRequest(BaseModel):
    image_base64: str
    question: str


class DetectResponse(BaseModel):
    objects: list[dict]
    latency_ms: float


class PointResponse(BaseModel):
    points: list[dict]
    latency_ms: float


class QueryResponse(BaseModel):
    answer: str
    latency_ms: float


def decode_image(b64: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


class LoRALinear(nn.Module):
    """LoRA adapter for a frozen Linear layer (inference only).

    Has an `enabled` flag to dynamically bypass LoRA (for detect/point
    which need the base model behavior).
    """

    enabled = True  # class-level toggle for all LoRA layers

    def __init__(self, base_linear, rank=8, alpha=16):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_features, dtype=base_linear.weight.dtype,
                        device=base_linear.weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=base_linear.weight.dtype,
                        device=base_linear.weight.device)
        )

    @property
    def weight(self):
        if not LoRALinear.enabled:
            return self.base.weight
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        base_out = self.base(x)
        if not LoRALinear.enabled:
            return base_out
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def inject_and_load_lora(model, lora_path: str, rank: int = 8):
    """Inject LoRA adapters and load trained weights."""
    inner = model.model
    lora_state = torch.load(lora_path, weights_only=True, map_location="cuda")
    alpha = rank * 2

    # Determine which blocks/layers need LoRA from the saved state
    blocks = inner.text["blocks"]
    lora_targets = ["qkv", "proj", "fc1", "fc2"]

    injected = 0
    for i, block in enumerate(blocks):
        for group_name in ["attn", "mlp"]:
            if group_name not in block:
                continue
            group = block[group_name]
            for layer_name in list(group.keys()):
                key_a = f"blocks.{i}.{group_name}.{layer_name}.lora_A"
                key_b = f"blocks.{i}.{group_name}.{layer_name}.lora_B"
                if key_a in lora_state and key_b in lora_state:
                    mod = group[layer_name]
                    if isinstance(mod, nn.Linear):
                        lora = LoRALinear(mod, rank=rank, alpha=alpha)
                        lora.lora_A = nn.Parameter(lora_state[key_a].to("cuda"))
                        lora.lora_B = nn.Parameter(lora_state[key_b].to("cuda"))
                        group[layer_name] = lora
                        injected += 1

    # lm_head
    if "lm_head.lora_A" in lora_state:
        lm = inner.text["lm_head"]
        if isinstance(lm, nn.Linear):
            lora = LoRALinear(lm, rank=rank, alpha=alpha)
            lora.lora_A = nn.Parameter(lora_state["lm_head.lora_A"].to("cuda"))
            lora.lora_B = nn.Parameter(lora_state["lm_head.lora_B"].to("cuda"))
            inner.text["lm_head"] = lora
            injected += 1

    logger.info(f"Injected and loaded {injected} LoRA adapters from {lora_path}")
    return injected


@app.on_event("startup")
def load_model():
    global model, device, lora_loaded
    logger.info("Loading Moondream 2B...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision=MD_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    torch.set_num_threads(4)
    device = "cuda"
    mem = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Model loaded in {time.time()-t0:.1f}s, VRAM: {mem:.1f}GB")

    # Load LoRA if specified
    if _args and _args.lora:
        lora_path = os.path.expanduser(_args.lora)
        if os.path.exists(lora_path):
            inject_and_load_lora(model, lora_path, rank=_args.lora_rank)
            lora_loaded = True
        else:
            logger.warning(f"LoRA weights not found at {lora_path}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "moondream2",
        "device": str(device),
        "lora_loaded": lora_loaded,
    }


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    image = decode_image(req.image_base64)
    # Disable LoRA for detect (uses base model's spatial understanding)
    LoRALinear.enabled = False
    try:
        enc = model.encode_image(image)
        t0 = time.time()
        result = model.detect(enc, req.query)
        latency_ms = (time.time() - t0) * 1000
    finally:
        LoRALinear.enabled = True

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
    # Disable LoRA for point (uses base model's spatial understanding)
    LoRALinear.enabled = False
    try:
        enc = model.encode_image(image)
        t0 = time.time()
        result = model.point(enc, req.query)
        latency_ms = (time.time() - t0) * 1000
    finally:
        LoRALinear.enabled = True

    points = result.get("points", [])
    # Add pixel coordinates
    w, h = image.size
    for pt in points:
        pt["pixel_x"] = int(pt["x"] * w)
        pt["pixel_y"] = int(pt["y"] * h)
        pt["image_w"] = w
        pt["image_h"] = h

    return PointResponse(points=points, latency_ms=latency_ms)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Query the model with a question about an image.

    Uses the fine-tuned model (with LoRA if loaded) to answer questions.
    For object-selection VLM: returns JSON like {"skill": "pick", "object": "red block"}.
    """
    image = decode_image(req.image_base64)
    t0 = time.time()
    result = model.query(image, req.question)
    latency_ms = (time.time() - t0) * 1000

    answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    return QueryResponse(answer=answer, latency_ms=latency_ms)


def check_port(port: int) -> bool:
    """Check if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def main():
    global _args
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to LoRA weights (best_lora.pt)")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank (must match training)")
    _args = parser.parse_args()

    if not check_port(_args.port):
        logger.error(f"Port {_args.port} already in use!")
        sys.exit(1)

    logger.info(f"Starting Moondream server on {_args.host}:{_args.port}")
    if _args.lora:
        logger.info(f"Will load LoRA from {_args.lora}")
    uvicorn.run(app, host=_args.host, port=_args.port, log_level="info")


if __name__ == "__main__":
    main()
