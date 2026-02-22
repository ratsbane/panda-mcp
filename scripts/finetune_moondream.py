#!/usr/bin/env python3
"""
Fine-tune Moondream 2B text decoder on robot pick/place skill data.

Run with: ~/panda-mcp/md_train_env/bin/python finetune_moondream.py

Uses LoRA adapters on the text model (~2M trainable params instead of 1.5B)
for better learning from small datasets (134 examples).

Freezes the vision encoder entirely. LoRA targets attention projections
in the text transformer.
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, random_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy VIPS logging
logging.getLogger("pyvips").setLevel(logging.WARNING)
for name in list(logging.Logger.manager.loggerDict):
    if "vips" in name.lower():
        logging.getLogger(name).setLevel(logging.WARNING)

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("VIPS_WARNING", "0")

MD_REVISION = "2025-01-09"


class SkillDataset(Dataset):
    def __init__(self, jsonl_path, image_dir):
        self.image_dir = Path(image_dir)
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        image = Image.open(self.image_dir / ex["image"]).convert("RGB")
        qa = ex["qa"][0]
        return image, qa["question"], qa["answer"]


def compute_loss(model, image, question, answer, device):
    """Teacher-forced forward pass through moondream's text decoder."""
    inner = model.model  # MoondreamModel inside HfMoondream

    from transformers_modules.vikhyatk.moondream2.adcbcd1a6d27fc19974b18dc128eb51ef6837879.text import (
        text_encoder,
    )

    # Step 1: Encode image (frozen, no grad)
    with torch.no_grad():
        encoded = inner.encode_image(image)
    kv_cache = encoded.kv_cache.clone()
    pos = encoded.pos

    # Step 2: Build token sequence
    q_ids = inner.tokenizer.encode(question).ids
    a_ids = inner.tokenizer.encode(answer).ids

    prefix = inner.config.tokenizer.templates["query"]["prefix"]
    suffix = inner.config.tokenizer.templates["query"]["suffix"]

    full_ids = prefix + q_ids + suffix + a_ids + [inner.config.tokenizer.eos_id]
    answer_start = len(prefix) + len(q_ids) + len(suffix)

    tokens = torch.tensor([full_ids], device=device)

    # Step 3: Forward through text model (WITH gradients on LoRA params)
    prompt_emb = text_encoder(tokens, inner.text)
    hidden = inner.ops["prefill"](
        prompt_emb, kv_cache, pos, inner.text, inner.config.text
    )

    # Apply lm_head projection to ALL positions
    lm_weight = inner.text["lm_head"].weight
    logits = hidden @ lm_weight.T  # (1, seq_len, vocab_size)

    if not hasattr(compute_loss, '_debug_done'):
        compute_loss._debug_done = True
        logger.info(f"  DEBUG: tokens shape={tokens.shape}, logits shape={logits.shape}")
        logger.info(f"  DEBUG: answer_start={answer_start}, total_len={len(full_ids)}")

    if logits.dim() == 3:
        logits = logits.squeeze(0)

    # Loss on answer tokens only, in fp32
    shift_logits = logits[answer_start - 1:-1, :].contiguous().float()
    shift_labels = tokens[0, answer_start:].contiguous()

    loss = nn.functional.cross_entropy(shift_logits, shift_labels)
    return loss


class FP32MasterWeightsOptimizer:
    """Maintains fp32 copies of fp16 parameters for stable training."""

    def __init__(self, fp16_params, lr, weight_decay=0.01):
        self.fp16_params = list(fp16_params)
        self.fp32_params = [p.float().detach().clone().requires_grad_(True)
                           for p in self.fp16_params]
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for fp16_p, fp32_p in zip(self.fp16_params, self.fp32_params):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float()
        self.optimizer.step()
        with torch.no_grad():
            for fp16_p, fp32_p in zip(self.fp16_params, self.fp32_params):
                fp16_p.copy_(fp32_p.half())

    def clip_grad_norm(self, max_norm):
        torch.nn.utils.clip_grad_norm_(self.fp32_params, max_norm)

    def set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr


def inject_lora(module_dict, target_keys, rank=16, alpha=32):
    """Inject LoRA adapters into Linear layers within a ModuleDict.

    For each target key matching a Linear layer, wraps it with a LoRA adapter.
    Returns list of LoRA parameter pairs (A, B) for the optimizer.
    """
    lora_params = []
    injected = 0

    for key in list(module_dict.keys()):
        mod = module_dict[key]
        if not isinstance(mod, nn.Linear):
            continue

        # Check if this key matches any target
        if not any(t in key for t in target_keys):
            continue

        # Create LoRA adapter wrapping the original Linear
        lora = LoRALinear(mod, rank=rank, alpha=alpha)
        module_dict[key] = lora
        lora_params.extend([lora.lora_A, lora.lora_B])
        injected += 1

    return lora_params, injected


class LoRALinear(nn.Module):
    """LoRA adapter for a frozen Linear layer."""

    def __init__(self, base_linear, rank=16, alpha=32):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # LoRA matrices — initialize A with kaiming, B with zeros
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, dtype=base_linear.weight.dtype,
                       device=base_linear.weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=base_linear.weight.dtype,
                       device=base_linear.weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze base
        base_linear.weight.requires_grad = False
        if base_linear.bias is not None:
            base_linear.bias.requires_grad = False

    @property
    def weight(self):
        """Return effective weight (for lm_head projection)."""
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
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info(f"Loading Moondream 2B (revision {MD_REVISION})...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision=MD_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    inner = model.model

    # Freeze ALL parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Inject LoRA into text model transformer blocks
    # Structure: text.blocks[i] -> ModuleDict with attn.{qkv,proj} and mlp.{fc1,fc2}
    lora_targets = ["qkv", "proj", "fc1", "fc2"]  # all linear layers in each block
    all_lora_params = []

    logger.info("Injecting LoRA adapters...")
    blocks = inner.text["blocks"]
    for i, block in enumerate(blocks):
        for group_name in ["attn", "mlp"]:
            if group_name not in block:
                continue
            group = block[group_name]
            for layer_name in list(group.keys()):
                mod = group[layer_name]
                if isinstance(mod, nn.Linear) and layer_name in lora_targets:
                    lora = LoRALinear(mod, rank=args.lora_rank, alpha=args.lora_rank * 2)
                    group[layer_name] = lora
                    all_lora_params.extend([lora.lora_A, lora.lora_B])
                    if i < 2 or i == len(blocks) - 1:  # log first 2 and last
                        logger.info(f"  LoRA on block[{i}].{group_name}.{layer_name}: {mod.in_features}x{mod.out_features}")

    # Also add LoRA to lm_head
    lm_head = inner.text["lm_head"]
    if isinstance(lm_head, nn.Linear):
        lora = LoRALinear(lm_head, rank=args.lora_rank, alpha=args.lora_rank * 2)
        inner.text["lm_head"] = lora
        all_lora_params.extend([lora.lora_A, lora.lora_B])
        logger.info(f"  LoRA on lm_head: {lm_head.in_features}x{lm_head.out_features}")

    trainable = sum(p.numel() for p in all_lora_params)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    logger.info(f"LoRA adapters: {len(all_lora_params)//2} layers (24 blocks x 4 + lm_head), rank={args.lora_rank}")

    if trainable == 0:
        logger.error("No LoRA adapters injected!")
        return

    # Dataset
    dataset = SkillDataset(
        os.path.join(args.dataset, "dataset.jsonl"), args.dataset
    )
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Split: {train_size} train, {val_size} val")

    # FP32 master weights optimizer (for stable fp16 LoRA training)
    optimizer = FP32MasterWeightsOptimizer(
        all_lora_params, lr=args.lr, weight_decay=0.01
    )
    logger.info(f"Optimizer: AdamW with fp32 master weights, lr={args.lr}")

    total_steps = len(train_ds) * args.epochs
    warmup_steps = min(len(train_ds) // 2, 30)

    def get_lr_scale(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0
    patience = 3
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_steps = 0
        skipped = 0

        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        for step, idx in enumerate(indices):
            image, question, answer = train_ds[idx]

            try:
                loss = compute_loss(model, image, question, answer, device)
            except Exception as e:
                if step < 5:
                    logger.warning(f"  Step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                skipped += 1
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                skipped += 1
                continue

            lr_scale = get_lr_scale(global_step)
            optimizer.set_lr(args.lr * lr_scale)

            optimizer.zero_grad()
            loss.backward()
            optimizer.clip_grad_norm(1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_steps += 1
            global_step += 1

            if step < 3:
                logger.info(f"  Step {step}: loss={loss.item():.4f} lr={args.lr * lr_scale:.2e}")

            if (step + 1) % 20 == 0:
                logger.info(
                    f"  Epoch {epoch+1} step {step+1}/{len(train_ds)} "
                    f"loss={epoch_loss/num_steps:.4f} lr={args.lr * lr_scale:.2e}"
                )

        avg_train = epoch_loss / max(num_steps, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for idx in range(len(val_ds)):
                image, question, answer = val_ds[idx]
                try:
                    loss = compute_loss(model, image, question, answer, device)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss += loss.item()
                        val_steps += 1
                except Exception:
                    continue

        avg_val = val_loss / max(val_steps, 1)
        logger.info(
            f"Epoch {epoch+1}/{args.epochs}: train={avg_train:.4f} val={avg_val:.4f} "
            f"(steps={num_steps}, skipped={skipped})"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Save LoRA weights only
            lora_state = {}
            blocks = inner.text["blocks"]
            for i, block in enumerate(blocks):
                for group_name in ["attn", "mlp"]:
                    if group_name not in block:
                        continue
                    group = block[group_name]
                    for layer_name in group.keys():
                        mod = group[layer_name]
                        if isinstance(mod, LoRALinear):
                            lora_state[f"blocks.{i}.{group_name}.{layer_name}.lora_A"] = mod.lora_A.data.cpu()
                            lora_state[f"blocks.{i}.{group_name}.{layer_name}.lora_B"] = mod.lora_B.data.cpu()
            lm = inner.text["lm_head"]
            if isinstance(lm, LoRALinear):
                lora_state["lm_head.lora_A"] = lm.lora_A.data.cpu()
                lora_state["lm_head.lora_B"] = lm.lora_B.data.cpu()

            torch.save(lora_state, output_dir / "best_lora.pt")
            logger.info(f"  Saved best LoRA ({len(lora_state)} tensors, val={avg_val:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping (no improvement for {patience} epochs)")
                break

    # Also save final full model for easy loading
    model.save_pretrained(output_dir / "final")
    logger.info(f"Saved final model to {output_dir / 'final'}")

    # Reload best LoRA weights before prediction
    best_path = output_dir / "best_lora.pt"
    if best_path.exists():
        logger.info("Reloading best LoRA weights...")
        lora_state = torch.load(best_path, weights_only=True)
        blocks = inner.text["blocks"]
        for key, val in lora_state.items():
            val = val.to(device)
            if key.startswith("blocks."):
                parts = key.split(".")
                i, group, layer, param = int(parts[1]), parts[2], parts[3], parts[4]
                mod = blocks[i][group][layer]
                if isinstance(mod, LoRALinear):
                    setattr(mod, param, nn.Parameter(val))
            elif key.startswith("lm_head."):
                param = key.split(".")[1]
                mod = inner.text["lm_head"]
                if isinstance(mod, LoRALinear):
                    setattr(mod, param, nn.Parameter(val))
        logger.info(f"Restored {len(lora_state)} LoRA tensors from best checkpoint")

    # Test predictions
    logger.info("\n=== Predictions ===")
    model.eval()
    for idx in range(min(5, len(val_ds))):
        image, question, answer = val_ds[idx]
        try:
            result = model.model.query(image, question)
            logger.info(f"  Q: {question}")
            logger.info(f"  Expected: {answer}")
            logger.info(f"  Got: {result['answer']}")
        except Exception as e:
            logger.info(f"  Error: {e}")

    summary = {
        "train_examples": train_size,
        "val_examples": val_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_params": trainable,
        "best_val_loss": best_val_loss,
        "final_train_loss": avg_train,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Done! {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.expanduser("~/panda-mcp/datasets/moondream_v1"))
    parser.add_argument("--output", default=os.path.expanduser("~/panda-mcp/models/moondream_lora"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()
    train(args)
