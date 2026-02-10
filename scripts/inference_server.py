#!/usr/bin/env python3
"""
SmolVLA inference server for Franka Panda pick-and-place.

Runs on DGX Spark (or any GPU machine). Loads a fine-tuned SmolVLA checkpoint
and serves action predictions via HTTP.

Usage:
    python inference_server.py --checkpoint ~/panda-vla-train/checkpoints/005000/pretrained_model --port 8085

Endpoints:
    POST /predict  - Predict action from observation (state + image + task)
    POST /reset    - Reset action queue (call at episode start)
    GET  /health   - Server status
"""

import argparse
import base64
import io
import logging
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SmolVLA Inference Server")

# Global state
policy = None
preprocessor = None
postprocessor = None
device = None
step_count = 0
start_time = None
inference_count = 0
last_task = None


class PredictRequest(BaseModel):
    state: list[float]  # 8 floats: 7 joints + gripper width
    image_b64: str  # base64-encoded JPEG
    task: str  # language instruction
    n_actions: int = 1  # number of actions to return (up to chunk_size)


class PredictResponse(BaseModel):
    actions: list[list[float]]  # list of 8-float action vectors
    step: int


class ResetRequest(BaseModel):
    task: str | None = None


def load_model(checkpoint_path: str, device_name: str):
    """Load SmolVLA policy and processors from checkpoint."""
    global policy, preprocessor, postprocessor, device, start_time

    logger.info(f"Loading SmolVLA from {checkpoint_path} on {device_name}")
    t0 = time.time()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    device = policy.config.device
    logger.info(f"Policy loaded on {device}")

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_path, "policy_preprocessor.json"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_path, "policy_postprocessor.json"
    )

    elapsed = time.time() - t0
    logger.info(f"Model loaded in {elapsed:.1f}s")
    start_time = time.time()


def decode_image(image_b64: str) -> torch.Tensor:
    """Decode base64 JPEG to float tensor (C, H, W) in [0, 1]."""
    jpeg_bytes = base64.b64decode(image_b64)
    jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)

    import cv2
    image_bgr = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode JPEG image")

    # BGR -> RGB, HWC -> CHW, uint8 -> float32 [0,1]
    image_rgb = image_bgr[:, :, ::-1].copy()
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    return tensor


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict next action(s) from observation.

    SmolVLA internally predicts a chunk of 50 actions at once. Only the first
    call runs the neural network; subsequent calls pop from a queue. Set
    n_actions > 1 to retrieve multiple actions in a single HTTP round-trip.
    """
    global step_count, inference_count, last_task

    if policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        t0 = time.time()
        n = min(max(req.n_actions, 1), 50)

        # Build observation dict
        image_tensor = decode_image(req.image_b64)
        state_tensor = torch.tensor(req.state, dtype=torch.float32)

        obs = {
            "observation.state": state_tensor,
            "observation.images.rgb": image_tensor,
            "task": req.task,
        }

        # Preprocess (tokenize language, normalize state, move to device)
        batch = preprocessor(obs)

        # Keep only tensor entries for select_action
        batch_clean = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Get N actions (first call runs model, rest pop from queue)
        actions_out = []
        with torch.no_grad():
            for _ in range(n):
                action = policy.select_action(batch_clean)
                post_result = postprocessor({"action": action.unsqueeze(0)})
                action_np = post_result["action"].squeeze().cpu().numpy()
                actions_out.append(action_np.tolist())

        elapsed_ms = (time.time() - t0) * 1000
        step_count += n
        inference_count += 1
        last_task = req.task

        logger.info(f"Predicted {n} actions in {elapsed_ms:.0f}ms (step {step_count})")

        return PredictResponse(
            actions=actions_out,
            step=step_count,
        )

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    """Reset action queue. Call at the start of each episode."""
    global step_count, last_task

    if policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    policy.reset()
    step_count = 0
    if req.task:
        last_task = req.task

    logger.info(f"Reset action queue (task: {req.task or last_task})")
    return {"success": True, "message": "Action queue reset"}


@app.get("/health")
async def health():
    """Server health check."""
    return {
        "status": "ok" if policy is not None else "no_model",
        "device": str(device) if device else None,
        "uptime_s": round(time.time() - start_time, 1) if start_time else 0,
        "inference_count": inference_count,
        "current_step": step_count,
        "last_task": last_task,
    }


def main():
    parser = argparse.ArgumentParser(description="SmolVLA Inference Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained_model directory",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    load_model(args.checkpoint, args.device)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
