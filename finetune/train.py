#!/usr/bin/env python3
"""
finetune/train.py
=================
Fine-tune Florence-2 on the generated MuJoCo scene dataset using LoRA.

Trains on both <OD> (object detection) and <DETAILED_CAPTION> tasks so the
model learns to recognise and describe MuJoCo scene objects (red_cube,
blue_cube, green_cylinder, etc.).

Usage
-----
    # Generate dataset first:
    python finetune/generate_dataset.py --mujoco-cli ../mujoco-cli

    # Fine-tune:
    python finetune/train.py

    # With options:
    python finetune/train.py --epochs 10 --lr 1e-4 --batch-size 4
    python finetune/train.py --model microsoft/Florence-2-base  # lighter
    python finetune/train.py --data finetune/data --output finetune/model

Prerequisites
-------------
    pip install peft>=0.11.0   # LoRA support
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Dataset ──────────────────────────────────────────────────────────────────

class Florence2Dataset(Dataset):
    """
    Dataset for Florence-2 fine-tuning.

    Each sample has:
      - image: PIL Image
      - prefix: task token (e.g. "<OD>" or "<DETAILED_CAPTION>")
      - suffix: expected model output
    """

    def __init__(self, jsonl_paths: list[str], data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = []
        for jsonl_path in jsonl_paths:
            p = Path(jsonl_path)
            if not p.is_absolute():
                p = self.data_dir / p.name
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.samples.append(json.loads(line))
        logger.info("Loaded %d samples from %d files", len(self.samples), len(jsonl_paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.data_dir / sample["image"]
        image = Image.open(image_path).convert("RGB")
        return {
            "image": image,
            "prefix": sample["prefix"],
            "suffix": sample["suffix"],
        }


def collate_fn(batch, processor, device, max_length=1024):
    """Collate a batch of samples for Florence-2 training."""
    images = [item["image"] for item in batch]
    prefixes = [item["prefix"] for item in batch]
    suffixes = [item["suffix"] for item in batch]

    # Resize images to square 768×768 (Florence-2 requirement)
    images = [img.resize((768, 768), Image.LANCZOS) for img in images]

    # Process inputs
    inputs = processor(
        text=prefixes,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Process labels (suffix = expected output)
    labels = processor.tokenizer(
        text=suffixes,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)

    return inputs, labels


# ── Training ─────────────────────────────────────────────────────────────────

def setup_lora(model, r: int = 8, alpha: int = 16):
    """Apply LoRA adapters to the model's language model layers."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("Error: peft not installed. Run: pip install peft>=0.11.0")
        sys.exit(1)

    # Target the language model's attention layers
    target_modules = []
    for name, _ in model.named_modules():
        if any(k in name for k in ["q_proj", "v_proj", "k_proj", "out_proj"]):
            if "language_model" in name:
                # Extract the module name relative to the model
                target_modules.append(name.split(".")[-1])
    # Deduplicate
    target_modules = list(set(target_modules))

    if not target_modules:
        # Fallback: target common attention layer names
        target_modules = ["q_proj", "v_proj"]

    logger.info("LoRA target modules: %s", target_modules)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def tie_weights(model):
    """Tie Florence-2 language model weights (same fix as analyzer.py)."""
    try:
        lm = model.language_model if hasattr(model, "language_model") else model.base_model.model.language_model
        shared = lm.model.shared
        lm.model.encoder.embed_tokens = shared
        lm.model.decoder.embed_tokens = shared
        lm.lm_head.weight = shared.weight
    except AttributeError:
        logger.warning("Could not tie weights — model structure may differ")


def train(
    model_name: str = "microsoft/Florence-2-large",
    data_dir: str = "finetune/data",
    output_dir: str = "finetune/model",
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 5e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    device: str = "auto",
    max_length: int = 1024,
):
    """Run the fine-tuning loop."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info("Device: %s", device)

    # Apply EncoderDecoderCache compatibility patch
    from vision.analyzer import _apply_encoder_decoder_cache_compat
    _apply_encoder_decoder_cache_compat()

    # Load model and processor
    logger.info("Loading Florence-2 from %s ...", model_name)
    import transformers.utils.logging as _hf_logging
    prev_level = _hf_logging.get_verbosity()
    _hf_logging.set_verbosity_error()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    finally:
        _hf_logging.set_verbosity(prev_level)

    tie_weights(model)
    model = model.to(device)

    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True
    )

    # Apply LoRA
    logger.info("Applying LoRA (r=%d, alpha=%d) ...", lora_r, lora_alpha)
    model = setup_lora(model, r=lora_r, alpha=lora_alpha)

    # Load dataset
    data_path = Path(data_dir)
    jsonl_files = list(data_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {data_dir}. Run generate_dataset.py first.")
        sys.exit(1)

    dataset = Florence2Dataset(
        jsonl_paths=[str(f) for f in jsonl_files],
        data_dir=data_dir,
    )
    logger.info("Dataset: %d samples", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor, device, max_length),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )

    # Training loop
    model.train()
    total_steps = len(dataloader) * epochs
    step = 0

    logger.info("Training: %d epochs, %d steps/epoch, %d total steps",
                epochs, len(dataloader), total_steps)

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for inputs, labels in dataloader:
            step += 1

            # Forward pass
            outputs = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                labels=labels,
            )
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if step % 10 == 0:
                avg = epoch_loss / n_batches
                print(
                    f"  Epoch {epoch+1}/{epochs}  "
                    f"Step {step}/{total_steps}  "
                    f"Loss: {loss.item():.4f}  "
                    f"Avg: {avg:.4f}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} — avg loss: {avg_loss:.4f}")

    # Save the fine-tuned model (LoRA adapters + processor)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(out_path))
    processor.save_pretrained(str(out_path))
    logger.info("Model saved to %s", out_path)

    print(f"\nFine-tuning complete!")
    print(f"  Model saved to: {out_path}")
    print(f"  To use: pass --model {out_path} to mujoco-cli-vision.py")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Florence-2 on MuJoCo scene objects"
    )
    parser.add_argument(
        "--model", default="microsoft/Florence-2-large",
        help="Base Florence-2 model (default: Florence-2-large)",
    )
    parser.add_argument(
        "--data", default="finetune/data",
        help="Dataset directory (default: finetune/data)",
    )
    parser.add_argument(
        "--output", default="finetune/model",
        help="Output directory for fine-tuned model (default: finetune/model)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    args = parser.parse_args()

    train(
        model_name=args.model,
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=args.device,
    )


if __name__ == "__main__":
    main()
