#!/usr/bin/env python3
"""
Visual grounding server using Qwen2.5-VL-3B.

Accepts image + text query, returns bounding box pixel coordinates.
Runs on DGX Spark as a FastAPI server.

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
        "confidence": 0.95,
        "raw_output": "..."
    }
"""

import argparse
import base64
import io
import json
import logging
import re
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grounding-server")

app = FastAPI(title="Visual Grounding Server")

# Global model/processor (loaded at startup)
model = None
processor = None


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


@app.on_event("startup")
async def load_model():
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


@app.post("/ground", response_model=GroundingResponse)
async def ground(req: GroundingRequest):
    """Find an object in the image given a natural language description."""
    if model is None:
        return GroundingResponse(success=False, error="Model not loaded")

    try:
        # Decode image
        img_data = base64.b64decode(req.image)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        orig_w, orig_h = image.size
        if req.width > 0:
            orig_w = req.width
        if req.height > 0:
            orig_h = req.height

        # Build the prompt for visual grounding
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            f"Point to {req.query}. "
                            "Output the bounding box coordinates as [x1, y1, x2, y2] "
                            "where coordinates are in the range 0-1000 (normalized). "
                            "Output ONLY the coordinates, nothing else."
                        ),
                    },
                ],
            }
        ]

        # Process with Qwen2.5-VL
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

        # Decode only the generated tokens
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        raw_output = processor.decode(generated, skip_special_tokens=True).strip()
        inference_ms = (time.time() - t0) * 1000

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


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
