#!/usr/bin/env python3
"""
Benchmark VLM models for zero-shot object pointing/grounding.

Tests each model's ability to locate colored blocks in a camera image
and output pixel coordinates, which are then converted to robot-frame
coordinates via homography for error measurement.

Usage:
    python benchmark_vlm_pointing.py /tmp/benchmark_frame.jpg

Requires: transformers, torch, Pillow
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


# Ground truth from color detection + ArUco homography (verified by 6/6 picks)
GROUND_TRUTH = {
    "green_block": {"pixel": (489, 453), "robot": (0.398, -0.178)},
    "red_block": {"pixel": (608, 432), "robot": (0.394, -0.096)},
    "blue_block": {"pixel": (783, 515), "robot": (0.473, 0.005)},
    "orange_block": {"pixel": (1090, 544), "robot": (0.521, 0.185)},
}

# Queries to test for each block
QUERIES = {
    "green_block": [
        "Point to the green block.",
        "Where is the green block in this image?",
    ],
    "red_block": [
        "Point to the red block on the white paper.",
        "Where is the red block in this image?",
    ],
    "blue_block": [
        "Point to the blue block.",
        "Where is the blue block in this image?",
    ],
    "orange_block": [
        "Point to the wooden block near the bottom right.",
        "Where is the tan/orange block near the right edge?",
    ],
}

IMAGE_W, IMAGE_H = 1280, 720


def pixel_error(pred_px, gt_px):
    """Euclidean distance in pixels."""
    return math.sqrt((pred_px[0] - gt_px[0])**2 + (pred_px[1] - gt_px[1])**2)


def parse_coordinates_from_text(text, image_w=IMAGE_W, image_h=IMAGE_H):
    """Try to extract (x, y) pixel coordinates from model output text.

    Handles various formats:
    - "point(0.5, 0.3)" or "<point>0.5, 0.3</point>" (normalized)
    - "(640, 360)" or "x=640, y=360" (absolute pixels)
    - "<loc_640><loc_360>" (Florence-2 style location tokens)
    - JSON: {"x": 640, "y": 360}
    """
    import re

    # Moondream point format: "x=0.5, y=0.3" or similar normalized coords
    # Sometimes outputs as "<point x=\"0.382\" y=\"0.641\" />"
    m = re.search(r'x["\s=:]+([0-9.]+)["\s,]+y["\s=:]+([0-9.]+)', text, re.IGNORECASE)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if x <= 1.0 and y <= 1.0:  # normalized
            return (int(x * image_w), int(y * image_h))
        return (int(x), int(y))

    # Normalized float pair like "(0.382, 0.641)" or "0.382, 0.641"
    m = re.search(r'[\(\[<]?\s*(0\.\d+)\s*[,;]\s*(0\.\d+)\s*[\)\]>]?', text)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        return (int(x * image_w), int(y * image_h))

    # Florence-2 location tokens: <loc_NNN> where NNN is 0-999 (normalized to 0-999)
    locs = re.findall(r'<loc_(\d+)>', text)
    if len(locs) >= 2:
        x = int(locs[0]) * image_w / 1000
        y = int(locs[1]) * image_h / 1000
        return (int(x), int(y))

    # Absolute pixel pair like "(640, 360)" or "640, 360"
    m = re.search(r'[\(\[]?\s*(\d{2,4})\s*[,;]\s*(\d{2,4})\s*[\)\]]?', text)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        if x <= image_w and y <= image_h:
            return (x, y)

    # JSON format
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            x = data.get("x", data.get("pixel_x"))
            y = data.get("y", data.get("pixel_y"))
            if x is not None and y is not None:
                return (int(x), int(y))
    except (json.JSONDecodeError, TypeError):
        pass

    return None


class ModelBenchmark:
    """Base class for VLM benchmarks."""
    name = "base"

    def setup(self):
        """Load model and processor."""
        raise NotImplementedError

    def query(self, image: Image.Image, prompt: str) -> dict:
        """Run a query and return results.

        Returns:
            dict with keys:
                - raw_output: str (model's raw text output)
                - parsed_pixel: tuple or None (extracted pixel coords)
                - latency_s: float (inference time)
        """
        raise NotImplementedError

    def cleanup(self):
        """Free model memory."""
        pass


class SmolVLM2Benchmark(ModelBenchmark):
    name = "SmolVLM2-500M"

    def setup(self):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        print(f"  Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).cuda()
        self.model.eval()
        print(f"  Loaded. Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def query(self, image, prompt):
        import torch

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        latency = time.time() - t0

        output = self.processor.decode(ids[0], skip_special_tokens=True)
        # Strip the input prompt from output
        if "assistant" in output.lower():
            output = output.split("assistant")[-1].strip()
        elif prompt in output:
            output = output[output.index(prompt) + len(prompt):].strip()

        return {
            "raw_output": output,
            "parsed_pixel": parse_coordinates_from_text(output),
            "latency_s": latency,
        }

    def cleanup(self):
        import torch, gc
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()


class Florence2Benchmark(ModelBenchmark):
    name = "Florence-2-base"

    def setup(self):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        model_id = "microsoft/Florence-2-base"
        print(f"  Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).cuda()
        self.model.eval()
        print(f"  Loaded. Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def query(self, image, prompt):
        import torch

        # Florence-2 uses task tokens for different capabilities
        # <CAPTION_TO_PHRASE_GROUNDING> for text-to-bbox grounding
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = self.processor(text=task + prompt, images=image, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=200,
                do_sample=False,
            )
        latency = time.time() - t0

        output = self.processor.decode(ids[0], skip_special_tokens=False)
        # Post-process Florence-2 output
        parsed = self.processor.post_process_generation(
            output, task=task, image_size=(image.width, image.height)
        )

        # Extract center of first bounding box if available
        pixel = None
        if task in parsed and "bboxes" in parsed[task]:
            bboxes = parsed[task]["bboxes"]
            if bboxes:
                x1, y1, x2, y2 = bboxes[0]
                pixel = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        return {
            "raw_output": str(parsed),
            "parsed_pixel": pixel,
            "latency_s": latency,
        }

    def cleanup(self):
        import torch, gc
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()


class Moondream2BBenchmark(ModelBenchmark):
    name = "Moondream-2B"

    def setup(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "vikhyatk/moondream2"
        print(f"  Loading {model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()
        print(f"  Loaded. Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def query(self, image, prompt):
        import torch

        t0 = time.time()

        # Moondream has a built-in point method for localization
        # First try the point API if the prompt asks for pointing
        try:
            # Use the model's built-in point method
            enc_image = self.model.encode_image(image)
            result = self.model.point(enc_image, prompt)
            latency = time.time() - t0

            # point() returns list of (x, y) normalized coordinates
            pixel = None
            raw = str(result)
            if result and len(result) > 0:
                pt = result[0]
                if hasattr(pt, 'x'):
                    pixel = (int(pt.x * image.width), int(pt.y * image.height))
                elif isinstance(pt, (list, tuple)) and len(pt) == 2:
                    x, y = pt
                    if x <= 1.0 and y <= 1.0:
                        pixel = (int(x * image.width), int(y * image.height))
                    else:
                        pixel = (int(x), int(y))

            return {
                "raw_output": raw,
                "parsed_pixel": pixel,
                "latency_s": latency,
            }
        except Exception as e:
            # Fallback to text generation
            enc_image = self.model.encode_image(image)
            answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
            latency = time.time() - t0

            return {
                "raw_output": answer,
                "parsed_pixel": parse_coordinates_from_text(answer),
                "latency_s": latency,
            }

    def cleanup(self):
        import torch, gc
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


class Moondream05BBenchmark(ModelBenchmark):
    name = "Moondream-0.5B"

    def setup(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "vikhyatk/moondream2"
        # The 0.5B model has a different revision/tag
        # Check if there's a specific 0.5B model
        model_id = "moondream/moondream-0_5b-int8-20250123"
        print(f"  Loading {model_id}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cuda",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"  Failed to load 0.5B model ({e}), skipping")
            raise
        self.model.eval()
        print(f"  Loaded. Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def query(self, image, prompt):
        import torch

        t0 = time.time()
        try:
            enc_image = self.model.encode_image(image)
            result = self.model.point(enc_image, prompt)
            latency = time.time() - t0

            pixel = None
            raw = str(result)
            if result and len(result) > 0:
                pt = result[0]
                if hasattr(pt, 'x'):
                    pixel = (int(pt.x * image.width), int(pt.y * image.height))
                elif isinstance(pt, (list, tuple)) and len(pt) == 2:
                    x, y = pt
                    if x <= 1.0 and y <= 1.0:
                        pixel = (int(x * image.width), int(y * image.height))
                    else:
                        pixel = (int(x), int(y))

            return {
                "raw_output": raw,
                "parsed_pixel": pixel,
                "latency_s": latency,
            }
        except Exception as e:
            enc_image = self.model.encode_image(image)
            answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
            latency = time.time() - t0

            return {
                "raw_output": answer,
                "parsed_pixel": parse_coordinates_from_text(answer),
                "latency_s": latency,
            }

    def cleanup(self):
        import torch, gc
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def run_benchmark(model_bench, image, results_all):
    """Run all queries against a single model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_bench.name}")
    print(f"{'='*60}")

    try:
        model_bench.setup()
    except Exception as e:
        print(f"  SETUP FAILED: {e}")
        results_all[model_bench.name] = {"error": str(e)}
        return

    results = {}
    for block_name, gt in GROUND_TRUTH.items():
        queries = QUERIES[block_name]
        block_results = []

        for query_text in queries:
            print(f"\n  Query: \"{query_text}\"")
            try:
                r = model_bench.query(image, query_text)
                print(f"  Raw output: {r['raw_output'][:200]}")
                print(f"  Parsed pixel: {r['parsed_pixel']}")
                print(f"  Latency: {r['latency_s']:.2f}s")

                if r["parsed_pixel"]:
                    px_err = pixel_error(r["parsed_pixel"], gt["pixel"])
                    print(f"  Pixel error: {px_err:.1f}px (vs gt {gt['pixel']})")
                else:
                    px_err = None
                    print(f"  Could not parse coordinates from output")

                block_results.append({
                    "query": query_text,
                    "raw_output": r["raw_output"],
                    "parsed_pixel": r["parsed_pixel"],
                    "pixel_error": px_err,
                    "latency_s": r["latency_s"],
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                block_results.append({
                    "query": query_text,
                    "error": str(e),
                })

        results[block_name] = block_results

    # Summary
    print(f"\n--- {model_bench.name} Summary ---")
    all_errors = []
    all_latencies = []
    for block_name, block_results in results.items():
        for r in block_results:
            if "pixel_error" in r and r["pixel_error"] is not None:
                all_errors.append(r["pixel_error"])
            if "latency_s" in r:
                all_latencies.append(r["latency_s"])

    if all_errors:
        print(f"  Pixel errors: mean={np.mean(all_errors):.1f}px, "
              f"median={np.median(all_errors):.1f}px, "
              f"max={np.max(all_errors):.1f}px")
        print(f"  Coordinate parse rate: {len(all_errors)}/{sum(len(QUERIES[b]) for b in GROUND_TRUTH)}")
    else:
        print(f"  No coordinates could be parsed from any output")

    if all_latencies:
        print(f"  Latency: mean={np.mean(all_latencies):.2f}s, "
              f"median={np.median(all_latencies):.2f}s")

    results_all[model_bench.name] = {
        "results": results,
        "summary": {
            "pixel_errors": all_errors,
            "mean_pixel_error": float(np.mean(all_errors)) if all_errors else None,
            "median_pixel_error": float(np.median(all_errors)) if all_errors else None,
            "max_pixel_error": float(np.max(all_errors)) if all_errors else None,
            "parse_rate": f"{len(all_errors)}/{sum(len(QUERIES[b]) for b in GROUND_TRUTH)}",
            "mean_latency_s": float(np.mean(all_latencies)) if all_latencies else None,
        }
    }

    model_bench.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLM pointing accuracy")
    parser.add_argument("image_path", help="Path to test image")
    parser.add_argument("--models", nargs="+",
                       default=["smolvlm2", "florence2", "moondream2b", "moondream05b"],
                       help="Models to benchmark")
    parser.add_argument("--output", default="/tmp/vlm_benchmark_results.json",
                       help="Output JSON path")
    args = parser.parse_args()

    image = Image.open(args.image_path).convert("RGB")
    print(f"Image: {image.size} from {args.image_path}")
    print(f"Ground truth blocks: {list(GROUND_TRUTH.keys())}")

    MODEL_MAP = {
        "smolvlm2": SmolVLM2Benchmark,
        "florence2": Florence2Benchmark,
        "moondream2b": Moondream2BBenchmark,
        "moondream05b": Moondream05BBenchmark,
    }

    results_all = {
        "image": args.image_path,
        "ground_truth": {k: {"pixel": list(v["pixel"]), "robot": list(v["robot"])}
                        for k, v in GROUND_TRUTH.items()},
        "models": {},
    }

    for model_key in args.models:
        if model_key not in MODEL_MAP:
            print(f"Unknown model: {model_key}")
            continue
        bench = MODEL_MAP[model_key]()
        run_benchmark(bench, image, results_all["models"])

    # Save results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(args.output, "w") as f:
        json.dump(results_all, f, indent=2, default=convert)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
