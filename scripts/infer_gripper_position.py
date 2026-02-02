#!/usr/bin/env python3
"""
Run inference with a trained gripper localizer model.

Usage:
    # Single image
    python scripts/infer_gripper_position.py --model runs/best/model_best.pt --image test.jpg

    # From camera (Pi)
    python scripts/infer_gripper_position.py --model runs/best/model_best.pt --camera

    # ONNX model (for AI Hat)
    python scripts/infer_gripper_position.py --onnx runs/best/model_best.onnx --camera
"""

import argparse
import time
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def load_pytorch_model(model_path: str, backbone: str = "mobilenetv3_large_100"):
    """Load a PyTorch model."""
    from train_gripper_localizer import GripperLocalizer

    model = GripperLocalizer(backbone=backbone, pretrained=False)

    # Load weights
    state_dict = torch.load(model_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_onnx_model(onnx_path: str):
    """Load an ONNX model for inference."""
    import onnxruntime as ort

    # Try to use GPU if available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    return session


def get_transform(img_size: int = 224):
    """Get inference transform."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_pytorch(model, image: Image.Image, transform, device):
    """Run prediction with PyTorch model."""
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.perf_counter()
        output = model(input_tensor)
        elapsed = time.perf_counter() - start

    position = output[0].cpu().numpy()
    return position, elapsed


def predict_onnx(session, image: Image.Image, transform):
    """Run prediction with ONNX model."""
    input_tensor = transform(image).unsqueeze(0).numpy()

    start = time.perf_counter()
    output = session.run(None, {"image": input_tensor})
    elapsed = time.perf_counter() - start

    position = output[0][0]
    return position, elapsed


def capture_from_camera():
    """Capture a frame from the camera daemon."""
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect('ipc:///tmp/camera-daemon.sock')
    sock.setsockopt(zmq.SUBSCRIBE, b'frame')
    sock.setsockopt(zmq.RCVTIMEO, 5000)

    topic, data, img_bytes = sock.recv_multipart()

    # Convert to PIL Image
    import io
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    sock.close()
    ctx.term()

    return image


def main():
    parser = argparse.ArgumentParser(description="Gripper position inference")
    parser.add_argument("--model", type=str, help="Path to PyTorch model")
    parser.add_argument("--onnx", type=str, help="Path to ONNX model")
    parser.add_argument("--backbone", type=str, default="mobilenetv3_large_100",
                        help="Backbone architecture (for PyTorch model)")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--camera", action="store_true", help="Capture from camera")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    args = parser.parse_args()

    if not args.model and not args.onnx:
        parser.error("Must specify --model or --onnx")

    if not args.image and not args.camera:
        parser.error("Must specify --image or --camera")

    # Load model
    if args.onnx:
        print(f"Loading ONNX model: {args.onnx}")
        model = load_onnx_model(args.onnx)
        use_onnx = True
    else:
        print(f"Loading PyTorch model: {args.model}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_pytorch_model(args.model, args.backbone).to(device)
        use_onnx = False
        print(f"Using device: {device}")

    transform = get_transform(args.img_size)

    # Benchmark mode
    if args.benchmark:
        print("\nRunning benchmark (100 iterations)...")

        # Get a test image
        if args.camera:
            image = capture_from_camera()
        else:
            image = Image.open(args.image).convert("RGB")

        # Warmup
        for _ in range(10):
            if use_onnx:
                predict_onnx(model, image, transform)
            else:
                predict_pytorch(model, image, transform, device)

        # Benchmark
        times = []
        for _ in range(100):
            if use_onnx:
                _, elapsed = predict_onnx(model, image, transform)
            else:
                _, elapsed = predict_pytorch(model, image, transform, device)
            times.append(elapsed)

        times = np.array(times) * 1000  # Convert to ms
        print(f"Latency: {np.mean(times):.2f}ms +/- {np.std(times):.2f}ms")
        print(f"Throughput: {1000 / np.mean(times):.1f} FPS")
        print(f"Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")
        return

    # Single or continuous inference
    while True:
        # Get image
        if args.camera:
            image = capture_from_camera()
        else:
            image = Image.open(args.image).convert("RGB")

        # Run inference
        if use_onnx:
            position, elapsed = predict_onnx(model, image, transform)
        else:
            position, elapsed = predict_pytorch(model, image, transform, device)

        # Print result
        print(f"Position: x={position[0]:.4f}m, y={position[1]:.4f}m, z={position[2]:.4f}m "
              f"({elapsed*1000:.1f}ms)")

        if not args.continuous:
            break

        time.sleep(0.1)  # Small delay for continuous mode


if __name__ == "__main__":
    main()
