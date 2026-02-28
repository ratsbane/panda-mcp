#!/usr/bin/env python3
"""
Visual grounding server using Qwen2.5-VL-3B.

Accepts image + text query, returns bounding box pixel coordinates.
Runs on DGX Spark as a FastAPI server (single-worker, serialized inference).

Usage:
    python scripts/grounding_server.py [--port 8090] [--host 0.0.0.0]

API:
    POST /ground
    {
        "image": "<base64 JPEG>",
        "query": "the red block",
        "width": 1280,   # original image width
        "height": 720    # original image height
    }
    Response:
    {
        "bbox": [x1, y1, x2, y2],   # pixel coordinates
        "center": [cx, cy],          # center point in pixels
        "raw_output": "..."
    }
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import time

# Limit PyTorch CPU threads BEFORE importing torch.
# Without this, torch spawns N threads per core (20 on Spark = 400+ threads
# under concurrent requests, which caused the "thread bomb" that killed Spark).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grounding-server")

# Global model/processor (loaded at startup)
model = None
processor = None

# Serialize inference — only one request at a time.
# Without this, concurrent requests each spawn full torch thread pools.
_inference_lock = asyncio.Lock()

# Max queued requests. If more pile up, reject immediately.
MAX_QUEUE_DEPTH = 3
_queue_depth = 0


class GroundingRequest(BaseModel):
    image: str  # base64-encoded JPEG
    query: str  # natural language reference
    width: int = 0  # original image width (for coordinate scaling)
    height: int = 0  # original image height


class GroundingResponse(BaseModel):
    success: bool
    bbox: list[int] | None = None  # [x1, y1, x2, y2] in pixel coords
    center: list[int] | None = None  # [cx, cy] center point
    raw_output: str = ""
    inference_ms: float = 0
    error: str | None = None


def parse_bbox_from_output(text: str, orig_w: int, orig_h: int) -> list[int] | None:
    """
    Parse bounding box coordinates from Qwen2.5-VL output.

    Qwen2.5-VL outputs bounding boxes in various formats:
    - JSON: {"bbox_2d": [x1, y1, x2, y2]}
    - Normalized coords (0-1000 scale): (123, 456), (789, 012)
    - Raw pixel coords
    """
    # Try JSON format first
    try:
        data = json.loads(text)
        if "bbox_2d" in data:
            return [int(c) for c in data["bbox_2d"]]
        if "bbox" in data:
            return [int(c) for c in data["bbox"]]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find coordinate patterns like (x1,y1),(x2,y2) or [x1,y1,x2,y2]
    # Qwen2.5-VL often outputs coordinates in 0-1000 normalized scale
    coord_patterns = [
        # (x1, y1), (x2, y2) format
        r'\((\d+),\s*(\d+)\).*?\((\d+),\s*(\d+)\)',
        # [x1, y1, x2, y2] format
        r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
        # x1, y1, x2, y2 as separate numbers in sequence
        r'(\d{2,4})[,\s]+(\d{2,4})[,\s]+(\d{2,4})[,\s]+(\d{2,4})',
    ]

    for pattern in coord_patterns:
        match = re.search(pattern, text)
        if match:
            coords = [int(match.group(i)) for i in range(1, 5)]
            # If coords are in 0-1000 normalized scale, convert to pixels
            if all(0 <= c <= 1000 for c in coords):
                coords = [
                    int(coords[0] * orig_w / 1000),
                    int(coords[1] * orig_h / 1000),
                    int(coords[2] * orig_w / 1000),
                    int(coords[3] * orig_h / 1000),
                ]
            return coords

    return None


def _run_inference(image: Image.Image, query: str, orig_w: int, orig_h: int) -> dict:
    """Run model inference synchronously. Called from executor thread."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"Point to {query}. "
                        "Output the bounding box coordinates as [x1, y1, x2, y2] "
                        "where coordinates are in the range 0-1000 (normalized). "
                        "Output ONLY the coordinates, nothing else."
                    ),
                },
            ],
        }
    ]

    t0 = time.time()
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    raw_output = processor.decode(generated, skip_special_tokens=True).strip()
    inference_ms = (time.time() - t0) * 1000

    return {"raw_output": raw_output, "inference_ms": inference_ms}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    global model, processor
    logger.info("Loading Qwen2.5-VL-3B-Instruct...")
    t0 = time.time()

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    elapsed = time.time() - t0
    logger.info(f"Model loaded in {elapsed:.1f}s")
    logger.info(f"Device: {model.device}")
    logger.info(f"Torch threads: {torch.get_num_threads()}, interop: {torch.get_num_interop_threads()}")

    yield

    logger.info("Shutting down, releasing model")
    model = None
    processor = None


app = FastAPI(title="Visual Grounding Server", lifespan=lifespan)


@app.post("/ground", response_model=GroundingResponse)
async def ground(req: GroundingRequest):
    """Find an object in the image given a natural language description."""
    global _queue_depth

    if model is None:
        return GroundingResponse(success=False, error="Model not loaded")

    # Reject if too many requests queued
    if _queue_depth >= MAX_QUEUE_DEPTH:
        return GroundingResponse(
            success=False,
            error=f"Server busy ({_queue_depth} requests queued, max {MAX_QUEUE_DEPTH})",
        )

    _queue_depth += 1
    try:
        # Decode image
        img_data = base64.b64decode(req.image)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        orig_w, orig_h = image.size
        if req.width > 0:
            orig_w = req.width
        if req.height > 0:
            orig_h = req.height

        # Serialize inference — one at a time
        async with _inference_lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, _run_inference, image, req.query, orig_w, orig_h
            )

        raw_output = result["raw_output"]
        inference_ms = result["inference_ms"]

        logger.info(f"Query: '{req.query}' -> '{raw_output}' ({inference_ms:.0f}ms)")

        # Parse bbox
        bbox = parse_bbox_from_output(raw_output, orig_w, orig_h)
        if bbox:
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            return GroundingResponse(
                success=True,
                bbox=bbox,
                center=[cx, cy],
                raw_output=raw_output,
                inference_ms=inference_ms,
            )
        else:
            return GroundingResponse(
                success=False,
                raw_output=raw_output,
                inference_ms=inference_ms,
                error=f"Could not parse bbox from model output: {raw_output}",
            )

    except Exception as e:
        logger.error(f"Grounding error: {e}", exc_info=True)
        return GroundingResponse(success=False, error=str(e))

    finally:
        _queue_depth -= 1


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "queue_depth": _queue_depth,
        "torch_threads": torch.get_num_threads(),
    }


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Visual grounding server (Qwen2.5-VL-3B)")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,         # Single worker — model is not fork-safe
        timeout_keep_alive=30,
        log_level="info",
    )
