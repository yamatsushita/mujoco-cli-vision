"""
vision/analyzer.py
==================
Scene analyzer powered by **Florence-2** (Microsoft, MIT licence).

Florence-2 is a unified vision-language foundation model (0.23 B / 0.77 B
parameters) that handles object detection, dense region captioning, phrase
grounding and more from a single, prompt-based interface.  It runs fully
locally — no API key required.

HuggingFace model cards:
  microsoft/Florence-2-base   (lighter, faster)
  microsoft/Florence-2-large  (recommended for accuracy)

Key tasks used here:
  <OD>                      – open-vocabulary object detection (boxes + labels)
  <DENSE_REGION_CAPTION>    – per-region natural-language captions
  <DETAILED_CAPTION>        – whole-image scene description
  <OPEN_VOCABULARY_DETECTION> – text-conditioned detection of named objects
"""

from __future__ import annotations

import glob as _glob
import io
import logging
import os
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)


# ── EncoderDecoderCache compatibility shim ────────────────────────────────────

def _apply_encoder_decoder_cache_compat() -> None:
    """
    Monkey-patch ``EncoderDecoderCache`` (transformers >= 5.x) to support the
    legacy integer-subscript indexing expected by Florence-2's remote model code.

    Florence-2's decoder iterates layers with ``past_key_values[idx]``, expecting
    a 4-tuple ``(self_key, self_val, cross_key, cross_val)`` per layer.  In
    transformers 5.x ``past_key_values`` is wrapped in an ``EncoderDecoderCache``
    object that does not support ``[]`` indexing.
    """
    try:
        from transformers.cache_utils import EncoderDecoderCache
    except ImportError:
        return

    if getattr(EncoderDecoderCache, "_florence2_compat_patched", False):
        return

    def _edc_getitem(self, idx):
        sac = self.self_attention_cache
        cac = self.cross_attention_cache
        if sac is None or not hasattr(sac, "key_cache") or idx >= len(sac.key_cache):
            return None
        self_kv = (sac.key_cache[idx], sac.value_cache[idx])
        if cac is not None and hasattr(cac, "key_cache") and idx < len(cac.key_cache):
            return self_kv + (cac.key_cache[idx], cac.value_cache[idx])
        return self_kv

    def _edc_len(self):
        sac = self.self_attention_cache
        if sac is not None and hasattr(sac, "key_cache"):
            return len(sac.key_cache)
        return 0

    EncoderDecoderCache.__getitem__ = _edc_getitem
    EncoderDecoderCache.__len__ = _edc_len
    EncoderDecoderCache._florence2_compat_patched = True


_apply_encoder_decoder_cache_compat()


# ── Florence-2 config cache patcher ──────────────────────────────────────────

def _patch_florence2_config_cache() -> bool:
    """
    Patch cached Florence-2 remote-code files to fix incompatibilities with
    transformers >= 5.x (tested with 5.5.4):

    1. ``configuration_florence2.py``:
       ``Florence2LanguageConfig.__init__`` accesses ``self.forced_bos_token_id``
       after ``super().__init__()``, but transformers 5.x no longer sets generation
       parameters on the config instance.  Fixed by using ``getattr`` instead.

    2. ``processing_florence2.py``:
       ``Florence2Processor.__init__`` reads ``tokenizer.additional_special_tokens``
       which is no longer an attribute in transformers 5.x.  Fixed by using
       ``getattr(tokenizer, 'additional_special_tokens', [])``.

    Returns True if at least one file was patched.
    """
    PATCHES = [
        # configuration_florence2.py — original bug (direct attribute access)
        (
            "configuration_florence2.py",
            "if self.forced_bos_token_id is None and kwargs.get(\"force_bos_token_to_be_generated\", False):",
            "if kwargs.get(\"force_bos_token_to_be_generated\", False):  # compat: skip forced_bos_token_id check",
        ),
        # configuration_florence2.py — partially-patched version (getattr still fails)
        (
            "configuration_florence2.py",
            "if getattr(self, 'forced_bos_token_id', None) is None and kwargs.get(\"force_bos_token_to_be_generated\", False):",
            "if kwargs.get(\"force_bos_token_to_be_generated\", False):  # compat: skip forced_bos_token_id check",
        ),
        # modeling_florence2.py — _supports_sdpa accessed before language_model is set
        (
            "modeling_florence2.py",
            "        return self.language_model._supports_sdpa",
            "        if not hasattr(self, 'language_model'):\n            return False\n        return self.language_model._supports_sdpa",
        ),
        # modeling_florence2.py — _supports_flash_attn_2 same issue
        (
            "modeling_florence2.py",
            "        return self.language_model._supports_flash_attn_2",
            "        if not hasattr(self, 'language_model'):\n            return False\n        return self.language_model._supports_flash_attn_2",
        ),
        # modeling_florence2.py — EncoderDecoderCache is not subscriptable in 5.x
        (
            "modeling_florence2.py",
            "            past_length = past_key_values[0][0].shape[2]",
            (
                "            # transformers >= 5.x uses EncoderDecoderCache instead of tuple-of-tuples\n"
                "            if hasattr(past_key_values, 'get_seq_length'):\n"
                "                past_length = past_key_values.get_seq_length()\n"
                "            elif hasattr(past_key_values, 'self_attention_cache'):\n"
                "                _sac = past_key_values.self_attention_cache\n"
                "                past_length = _sac.get_seq_length() if hasattr(_sac, 'get_seq_length') else _sac.key_cache[0].shape[2]\n"
                "            else:\n"
                "                past_length = past_key_values[0][0].shape[2]"
            ),
        ),
        # processing_florence2.py — additional_special_tokens removed in 5.x
        (
            "processing_florence2.py",
            "tokenizer.additional_special_tokens +",
            "getattr(tokenizer, 'additional_special_tokens', []) +",
        ),
    ]

    # Resolve HuggingFace cache root (respects HF_HOME / HF_HUB_CACHE env vars)
    hf_cache = Path(
        os.environ.get("HF_HOME", "")
        or os.environ.get("HF_HUB_CACHE", "")
        or os.environ.get("HUGGINGFACE_HUB_CACHE", "")
        or (Path.home() / ".cache" / "huggingface")
    )
    modules_dir = hf_cache / "modules" / "transformers_modules"

    patched_any = False
    for filename, BAD, GOOD in PATCHES:
        for path_str in _glob.glob(
            str(modules_dir / "**" / filename), recursive=True
        ):
            cached_file = Path(path_str)
            try:
                text = cached_file.read_text(encoding="utf-8")
            except OSError:
                continue

            if BAD not in text or GOOD in text:
                continue  # Not buggy or already patched

            cached_file.write_text(text.replace(BAD, GOOD), encoding="utf-8")
            logger.info("Patched Florence-2 cached file: %s", cached_file)
            patched_any = True

            # Remove compiled bytecode so Python recompiles from patched source
            pycache = cached_file.parent / "__pycache__"
            if pycache.exists():
                stem = cached_file.stem
                for pyc in pycache.glob(f"{stem}*.pyc"):
                    try:
                        pyc.unlink()
                    except OSError:
                        pass

    # Evict any partially loaded module entries from Python's import cache
    for key in list(sys.modules):
        if "configuration_florence2" in key or "processing_florence2" in key or "florence2" in key.lower():
            del sys.modules[key]

    return patched_any


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DetectedObject:
    """A single object detected in the scene."""
    label: str
    bbox_pixels: list[float]          # [x1, y1, x2, y2] in image pixels
    center_norm: list[float]          # [cx, cy] normalised to [0, 1]
    area_norm: float                  # fraction of image area
    caption: Optional[str] = None     # rich description if available

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bbox_pixels": self.bbox_pixels,
            "center_norm": self.center_norm,
            "area_norm": self.area_norm,
            "caption": self.caption,
        }


@dataclass
class SceneAnalysis:
    """Result of a full scene analysis."""
    caption: str
    objects: list[DetectedObject] = field(default_factory=list)
    image_size: list[int] = field(default_factory=lambda: [0, 0])   # [W, H]

    def to_dict(self) -> dict:
        return {
            "caption": self.caption,
            "objects": [o.to_dict() for o in self.objects],
            "image_size": self.image_size,
        }

    def to_context_string(self) -> str:
        """
        Produce a Markdown block for the REST API / interactive display.
        """
        lines = [
            "## Scene perception (auto-generated by vision module)",
            "",
            f"**Scene overview:** {self.caption}",
            "",
            "**Detected objects** (image coordinates: origin = top-left,"
            " values normalised to 0–1):",
        ]
        if self.objects:
            for i, obj in enumerate(self.objects, start=1):
                cx, cy = obj.center_norm
                area_pct = obj.area_norm * 100
                lines.append(
                    f"  {i:2d}. `{obj.label}` — centre ({cx:.2f}, {cy:.2f}),"
                    f" area {area_pct:.1f}%"
                    + (f"\n       ↳ {obj.caption}" if obj.caption else "")
                )
        else:
            lines.append("  _(no objects detected)_")
        lines += ["", "---", ""]
        return "\n".join(lines)

    def to_scene_text(self) -> str:
        """
        Plain-text scene description for embedding in a planner prompt.

        Produces a compact, line-oriented block that reads naturally inside
        the ``Current scene state:`` section of the mujoco-cli prompt templates.
        Image coordinates are normalised to [0, 1] with origin at the
        top-left corner of the rendered frame.
        """
        lines = [
            "=== Visual Scene (Florence-2 perception) ===",
            f"Scene overview: {self.caption}",
            "",
            "Detected objects (image coords, origin=top-left, values 0-1):",
        ]
        if self.objects:
            for i, obj in enumerate(self.objects, start=1):
                cx, cy = obj.center_norm
                area_pct = obj.area_norm * 100
                extra = f" — {obj.caption}" if obj.caption and obj.caption != obj.label else ""
                lines.append(
                    f"  {i:2d}. {obj.label!r} at image centre"
                    f" ({cx:.2f}, {cy:.2f}), area {area_pct:.1f}%{extra}"
                )
        else:
            lines.append("  (no objects detected)")
        lines.append("")
        return "\n".join(lines)


# ── Main analyzer ─────────────────────────────────────────────────────────────

class SceneAnalyzer:
    """
    Wraps Florence-2 for MuJoCo scene understanding.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Use ``"microsoft/Florence-2-base"`` for lower
        memory footprint or ``"microsoft/Florence-2-large"`` for best accuracy.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU.
        Pass ``"cpu"``, ``"cuda"``, or ``"mps"`` to override.
    dtype:
        Torch dtype.  ``None`` → float16 on GPU, float32 on CPU.
    """

    DEFAULT_MODEL = "microsoft/Florence-2-large"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
    ):
        self.device = self._resolve_device(device)
        self.dtype = dtype or (
            torch.float16 if self.device != "cpu" else torch.float32
        )
        logger.info("Loading Florence-2 from %s on %s …", model_name, self.device)
        self.model = self._load_model(model_name)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Florence-2 ready.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(spec: str) -> str:
        if spec == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return spec

    def _load_model(self, model_name: str):
        """
        Load the Florence-2 model, auto-patching the HuggingFace modules cache
        for known incompatibilities with transformers >= 5.x.

        Strategy:
          1. Try ``from_pretrained`` with ``attn_implementation="eager"`` (bypasses
             the ``_supports_sdpa`` attribute check added in transformers 5.x).
          2. If an ``AttributeError`` related to ``forced_bos_token_id`` or
             ``additional_special_tokens`` is raised, patch the cached remote-code
             files and retry once.
        """
        def _do_load():
            # Suppress the spurious "MISSING" load report for tied weights
            # (encoder/decoder embed_tokens + lm_head share the 'shared' embedding;
            # they are not stored separately in the checkpoint).
            import transformers.utils.logging as _hf_logging
            prev_level = _hf_logging.get_verbosity()
            _hf_logging.set_verbosity_error()
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=self.dtype,
                    trust_remote_code=True,
                    attn_implementation="eager",
                ).to(self.device)
            finally:
                _hf_logging.set_verbosity(prev_level)

            # transformers >= 5.x may not auto-tie weights for custom remote-code
            # models.  Manually tie the language model head to shared embeddings so
            # the model produces coherent output.
            try:
                shared = model.language_model.model.shared
                model.language_model.model.encoder.embed_tokens = shared
                model.language_model.model.decoder.embed_tokens = shared
                model.language_model.lm_head.weight = shared.weight
                logger.debug("Tied Florence-2 language-model weights to shared embedding.")
            except AttributeError as tie_exc:
                logger.warning("Could not tie Florence-2 weights: %s", tie_exc)
            return model

        try:
            return _do_load()
        except AttributeError as exc:
            msg = str(exc)
            _known = ("forced_bos_token_id", "additional_special_tokens",
                      "_supports_sdpa", "_supports_flash_attn_2", "EncoderDecoderCache")
            if not any(k in msg for k in _known):
                raise
            logger.warning(
                "Florence-2 remote-code incompatibility detected (%s). "
                "Patching cached files and retrying…",
                exc,
            )
            if not _patch_florence2_config_cache():
                logger.info("No cached files needed patching (may already be patched).")
            return _do_load()

    def _run_task(
        self,
        image: Image.Image,
        task_token: str,
        text_input: str = "",
        max_new_tokens: int = 1024,
    ) -> dict:
        """Run a single Florence-2 task and return the post-processed dict."""
        # Florence-2's DaViT vision tower requires square feature maps, and
        # CLIPImageProcessor in transformers >= 5.x may not auto-resize.
        # Resize to the processor's expected 768×768 input size.
        orig_w, orig_h = image.width, image.height
        proc_size = getattr(self.processor, 'image_processor', None)
        target = 768
        if proc_size and hasattr(proc_size, 'size'):
            sz = proc_size.size
            if isinstance(sz, dict):
                target = sz.get('height', 768)
            elif isinstance(sz, int):
                target = sz
        image = image.resize((target, target), Image.LANCZOS)

        prompt = task_token if not text_input else f"{task_token} {text_input}"
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        # Move all tensor inputs to device/dtype
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        return self.processor.post_process_generation(
            generated_text,
            task=task_token,
            image_size=(orig_w, orig_h),
        )

    @staticmethod
    def _bbox_to_object(
        bbox: list[float],
        label: str,
        img_w: int,
        img_h: int,
        caption: Optional[str] = None,
    ) -> DetectedObject:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        area = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
        return DetectedObject(
            label=label,
            bbox_pixels=bbox,
            center_norm=[round(cx, 4), round(cy, 4)],
            area_norm=round(area, 4),
            caption=caption,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def caption(self, image: Image.Image) -> str:
        """Return a single-sentence detailed description of the whole image."""
        result = self._run_task(image, "<DETAILED_CAPTION>")
        return result.get("<DETAILED_CAPTION>", "").strip()

    def detect_objects(self, image: Image.Image) -> list[DetectedObject]:
        """
        Open-vocabulary object detection.

        Uses the ``<OD>`` task which returns bounding boxes and class labels
        for all prominent objects Florence-2 can identify.
        """
        result = self._run_task(image, "<OD>")
        od = result.get("<OD>", {})
        bboxes = od.get("bboxes", [])
        labels = od.get("labels", [])
        return [
            self._bbox_to_object(bbox, lbl, image.width, image.height)
            for bbox, lbl in zip(bboxes, labels)
        ]

    def dense_captions(self, image: Image.Image) -> list[DetectedObject]:
        """
        Dense region captioning: returns per-region natural language captions
        together with their bounding boxes.  Richer than ``detect_objects``
        but slower.
        """
        result = self._run_task(image, "<DENSE_REGION_CAPTION>")
        drc = result.get("<DENSE_REGION_CAPTION>", {})
        bboxes = drc.get("bboxes", [])
        labels = drc.get("labels", [])
        return [
            self._bbox_to_object(bbox, lbl, image.width, image.height, caption=lbl)
            for bbox, lbl in zip(bboxes, labels)
        ]

    def ground_objects(
        self, image: Image.Image, text_query: str
    ) -> list[DetectedObject]:
        """
        Text-conditioned open-vocabulary detection.

        Finds instances of objects described by ``text_query``
        (e.g. ``"red cube. blue sphere. robot gripper"``).

        Use ``.`` or ``|`` as separators for multiple object types.
        """
        result = self._run_task(
            image, "<OPEN_VOCABULARY_DETECTION>", text_query
        )
        ovd = result.get("<OPEN_VOCABULARY_DETECTION>", {})
        bboxes = ovd.get("bboxes", [])
        labels = ovd.get("labels", [])
        return [
            self._bbox_to_object(bbox, lbl, image.width, image.height)
            for bbox, lbl in zip(bboxes, labels)
        ]

    def analyze_scene(
        self,
        image: Image.Image,
        use_dense_captions: bool = False,
    ) -> SceneAnalysis:
        """
        Full scene analysis pipeline.

        1. Whole-image caption  (``<DETAILED_CAPTION>``)
        2. Object detection     (``<OD>`` or ``<DENSE_REGION_CAPTION>``)

        Parameters
        ----------
        use_dense_captions:
            When True, uses ``<DENSE_REGION_CAPTION>`` which is slower but
            provides richer per-object descriptions.
        """
        caption = self.caption(image)
        objects = (
            self.dense_captions(image)
            if use_dense_captions
            else self.detect_objects(image)
        )
        return SceneAnalysis(
            caption=caption,
            objects=objects,
            image_size=[image.width, image.height],
        )

    # ── Convenience: load image from various sources ──────────────────────────

    @staticmethod
    def load_image(source) -> Image.Image:
        """
        Accept a file path, bytes, BytesIO, or PIL Image and return RGB PIL Image.
        """
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")
        if isinstance(source, (bytes, bytearray)):
            return Image.open(io.BytesIO(source)).convert("RGB")
        if isinstance(source, io.IOBase):
            return Image.open(source).convert("RGB")
        raise TypeError(f"Unsupported image source type: {type(source)}")
