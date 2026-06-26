"""
OWLv2 + DINOv2 few-shot detection (per-crop pipeline).

Step 0 — Source-image context filter: resolve each crop's raw source image,
         embed with DINOv2, keep only crops whose source ≈ target scene.
Step 1 — Per-crop loop: for each surviving query crop individually:
           OWLv2 single-query proposals → DINOv2 score each proposal
           against that same crop (no averaged prototype).
Step 2 — Cross-crop NMS merges overlapping detections from all crops.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import (
    Owlv2ForObjectDetection, Owlv2Processor,
    AutoImageProcessor, AutoModel,
)


# --- Config ---
OWL_MODEL_ID      = "google/owlv2-base-patch16-ensemble"
DINO_MODEL_ID     = "facebook/dinov2-base"
DEFAULT_OWL_THRESHOLD   = 0.2   # low — cast wide net, DINOv2 will filter
DEFAULT_NMS_THRESHOLD   = 0.1
DEFAULT_DINO_THRESHOLD  = 0.6   # cosine similarity cutoff
DEFAULT_SOURCE_SIM      = 0.75   # source-image-to-target similarity cutoff
DEFAULT_MAX_QUERIES     = 30
MODEL_NATIVE_RES        = 640
MIN_CONTENT_RATIO       = 0.10
DINO_INPUT_SIZE         = 224
IMG_EXTS                = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_STEM_RE            = re.compile(r"^(.+)_cls\d+_\d+$")

def parse_args():
    p = argparse.ArgumentParser(description="OWLv2 + DINOv2 few-shot detection")
    p.add_argument("--queries",          required=True,              help="Folder of query crop images")
    p.add_argument("--target",           required=True,              help="Target image to detect in")
    p.add_argument("--source-images",    required=True,               help="Folder of raw source images (enables context-aware query filtering)")
    p.add_argument("--source-sim",       type=float, default=DEFAULT_SOURCE_SIM,   help="Source-to-target similarity threshold (default: 0.5)")
    p.add_argument("--owl-threshold",    type=float, default=DEFAULT_OWL_THRESHOLD,  help="OWLv2 score threshold (default: 0.2)")
    p.add_argument("--nms-threshold",    type=float, default=DEFAULT_NMS_THRESHOLD,  help="NMS IoU threshold (default: 0.3)")
    p.add_argument("--dino-threshold",   type=float, default=DEFAULT_DINO_THRESHOLD, help="DINOv2 cosine similarity cutoff (default: 0.8)")
    p.add_argument("--max-queries",      type=int,   default=DEFAULT_MAX_QUERIES,    help="Max query crops to load (default: 10)")
    p.add_argument("--dtype",            default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--show",             action="store_true", help="Show result with matplotlib")
    return p.parse_args()


# --- DINOv2 ---

def dino_similarities_batch(
    crops: list[Image.Image],
    prototype: torch.Tensor,
    dino_processor: AutoImageProcessor,
    dino_model: AutoModel,
    device: str,
    batch_size: int = 16,
) -> list[float]:
    """Cosine similarity of all crops vs prototype in batched forward passes."""
    sims = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        inputs = dino_processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
        cls_embeds = F.normalize(outputs.last_hidden_state[:, 0], dim=-1)  # (B, hidden)
        batch_sims = F.cosine_similarity(cls_embeds, prototype.expand(len(batch), -1))
        sims.extend(batch_sims.tolist())
    return sims


# --- Source-image context filtering ---

def dino_embed_image(
    image: Image.Image,
    dino_processor: AutoImageProcessor,
    dino_model: AutoModel,
    device: str,
) -> torch.Tensor:
    """Return L2-normalised CLS embedding for a single image. Shape (1, hidden)."""
    inputs = dino_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = dino_model(**inputs)
    return F.normalize(out.last_hidden_state[:, 0], dim=-1)


def resolve_source_image(crop_path: Path, source_dir: Path) -> Path | None:
    """Map  <stem>_cls<id>_<idx>.png  →  <stem>.{jpg,png,...} in source_dir."""
    m = CROP_STEM_RE.match(crop_path.stem)
    if not m:
        return None
    src_stem = m.group(1)
    for ext in IMG_EXTS:
        candidate = source_dir / (src_stem + ext)
        if candidate.exists():
            return candidate
    return None


def filter_crops_by_source_similarity(
    crop_paths: list[Path],
    source_dir: Path,
    target_image: Image.Image,
    dino_processor: AutoImageProcessor,
    dino_model: AutoModel,
    device: str,
    threshold: float = DEFAULT_SOURCE_SIM,
) -> list[tuple[Path, float]]:
    """
    Keep only crops whose source image is similar to the target.
    Returns list of (crop_path, source_sim) for crops that pass.
    """
    target_emb = dino_embed_image(target_image, dino_processor, dino_model, device)

    # cache: source_stem → similarity (avoid re-embedding the same source image)
    source_sim_cache: dict[str, float] = {}
    kept: list[tuple[Path, float]] = []

    for cp in crop_paths:
        src_path = resolve_source_image(cp, source_dir)
        if src_path is None:
            print(f"[source-filter] Could not resolve source for {cp.name} — skipping")
            continue

        stem = src_path.stem
        if stem not in source_sim_cache:
            src_img = Image.open(src_path).convert("RGB")
            src_emb = dino_embed_image(src_img, dino_processor, dino_model, device)
            sim = float(F.cosine_similarity(src_emb, target_emb).item())
            source_sim_cache[stem] = sim

        sim = source_sim_cache[stem]
        if sim >= threshold:
            kept.append((cp, sim))
            print(f"[source-filter] {cp.name} — source sim {sim:.3f} ✓")
        else:
            print(f"[source-filter] {cp.name} — source sim {sim:.3f} < {threshold} — dropped")

    # summary: unique source images and their sims
    print(f"[source-filter] Unique source sims: "
          + ", ".join(f"{s}={v:.3f}" for s, v in sorted(source_sim_cache.items(), key=lambda x: -x[1])))
    print(f"[source-filter] {len(kept)}/{len(crop_paths)} crop(s) pass "
          f"(source-target sim ≥ {threshold})")
    return kept


# --- OWLv2 ---

def run_owlv2_single(
    processor: Owlv2Processor,
    model: Owlv2ForObjectDetection,
    query_image: Image.Image,
    target_image: Image.Image,
    threshold: float,
    nms_threshold: float,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> list[dict]:
    """OWLv2 image-guided detection with a SINGLE query crop. Returns list of {box, score} dicts."""
    inputs = processor(
        images=target_image,
        query_images=query_image,
        return_tensors="pt",
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)

    h, w = target_image.size[1], target_image.size[0]
    target_sizes = torch.tensor([[h, w]], device=device)
    results = processor.post_process_image_guided_detection(
        outputs=outputs,
        threshold=threshold,
        nms_threshold=nms_threshold,
        target_sizes=target_sizes,
    )

    boxes, scores = [], []
    for box, score in zip(results[0]["boxes"], results[0]["scores"]):
        boxes.append(box)
        scores.append(score)

    if not boxes:
        return []

    boxes_t  = torch.stack(boxes)
    scores_t = torch.stack(scores)
    keep     = torch.ops.torchvision.nms(boxes_t, scores_t, nms_threshold)

    return [
        {"box": [round(v, 1) for v in boxes_t[i].tolist()], "score": round(float(scores_t[i]), 4)}
        for i in keep
    ]


# --- Helpers ---

def _content_ratio(img: Image.Image) -> float:
    arr = np.asarray(img)
    return float((arr.max(axis=-1) > 10).mean())


def load_query_crops(
    query_dir: Path, max_queries: int = DEFAULT_MAX_QUERIES,
) -> tuple[list[Path], list[Image.Image]]:
    """Load crop images, return (paths, PIL images) — paths needed for source resolution."""
    exts  = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in query_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No images found in {query_dir}")

    kept_paths, crops = [], []
    for p in paths:
        img   = Image.open(p).convert("RGB")
        ratio = _content_ratio(img)
        if ratio < MIN_CONTENT_RATIO:
            print(f"[query] Skipping {p.name} — content ratio {ratio:.2%} (too blank)")
            continue
        kept_paths.append(p)
        crops.append(img)
        if len(crops) == max_queries:
            break

    if not crops:
        raise ValueError("All query crops rejected (too blank).")
    print(f"[query] Loaded {len(crops)} crop(s) from {query_dir}")
    return kept_paths, crops


def visualize(target_image: Image.Image, detections: list[dict], dino_threshold: float):
    _, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(target_image)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5,
            f"owl={det['owl_score']:.2f} dino={det['dino_sim']:.2f}",
            color="lime", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    ax.set_title(f"OWLv2+DINOv2 (dino_threshold={dino_threshold}) — {len(detections)} found")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# --- Main ---
def main():
    args = parse_args()

    query_dir      = Path(args.queries)
    target_path    = Path(args.target)
    source_dir     = Path(args.source_images)
    source_sim_thr = args.source_sim
    owl_threshold  = args.owl_threshold
    nms_threshold  = args.nms_threshold
    dino_threshold = args.dino_threshold
    max_queries    = args.max_queries
    dtype_str      = args.dtype
    show           = args.show
    device         = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype     = dtype_map[dtype_str]
    if dtype != torch.float32 and device == "cpu":
        print(f"[warn] {dtype_str} not supported on CPU — falling back to fp32")
        dtype = torch.float32

    print(f"[device] {device}  dtype: {dtype_str}")

    def _load(fn, *args, **kwargs):
        """Try normal load first; fall back to cache-only if network is unavailable."""
        try:
            return fn(*args, **kwargs)
        except Exception:
            return fn(*args, local_files_only=True, **kwargs)

    print(f"[model]  Loading OWLv2: {OWL_MODEL_ID} ...")
    owl_processor = _load(Owlv2Processor.from_pretrained, OWL_MODEL_ID)
    owl_model     = _load(Owlv2ForObjectDetection.from_pretrained, OWL_MODEL_ID).to(device=device, dtype=dtype)
    owl_model.eval()

    print(f"[model]  Loading DINOv2: {DINO_MODEL_ID} ...")
    dino_processor = _load(AutoImageProcessor.from_pretrained, DINO_MODEL_ID)
    dino_model     = _load(AutoModel.from_pretrained, DINO_MODEL_ID).to(device)
    dino_model.eval()

    query_paths, query_images = load_query_crops(query_dir, max_queries)

    target_original = Image.open(target_path).convert("RGB")
    print(f"[target] {target_path.name}  original={target_original.size}")

    # always resize for OWLv2 (small target → large objects become findable)
    if max(target_original.size) > MODEL_NATIVE_RES:
        owl_scale = MODEL_NATIVE_RES / max(target_original.size)
        owl_size  = (int(target_original.width * owl_scale), int(target_original.height * owl_scale))
        target_owl = target_original.resize(owl_size, Image.LANCZOS)
        print(f"[target] OWLv2 input resized → {target_owl.size} (scale={owl_scale:.3f})")
    else:
        target_owl = target_original
        owl_scale  = 1.0

    # Step 0 — context-aware query filtering
    if not source_dir.is_dir():
        print(f"[error] Source images folder not found: {source_dir}")
        return
    kept = filter_crops_by_source_similarity(
        query_paths, source_dir, target_original,
        dino_processor, dino_model, device,
        threshold=source_sim_thr,
    )
    if not kept:
        print("[error] No query crops passed source-similarity filter — nothing to detect")
        return
    query_paths  = [cp for cp, _ in kept]
    query_images = [Image.open(cp).convert("RGB") for cp in query_paths]

    # Step 1+2 — per-crop: OWLv2 proposals → DINOv2 filter against that same crop
    all_detections = []  # collect across all query crops
    for qi, (qpath, qimg) in enumerate(zip(query_paths, query_images)):
        # embed this single query crop as the reference
        query_emb = dino_embed_image(qimg, dino_processor, dino_model, device)

        # OWLv2 with this one crop on resized target
        proposals = run_owlv2_single(
            owl_processor, owl_model, qimg, target_owl,
            threshold=owl_threshold, nms_threshold=nms_threshold,
            device=device, dtype=dtype,
        )
        if not proposals:
            print(f"[crop {qi}] {qpath.name} — 0 proposals")
            continue

        # scale boxes back to original resolution, crop from original for DINOv2
        orig_w, orig_h = target_original.size
        valid_props, valid_crops = [], []
        for prop in proposals:
            # scale from owl-resized coords → original coords
            ox1 = prop["box"][0] / owl_scale
            oy1 = prop["box"][1] / owl_scale
            ox2 = prop["box"][2] / owl_scale
            oy2 = prop["box"][3] / owl_scale
            bw, bh = ox2 - ox1, oy2 - oy1
            # 10% padding
            x1 = max(0, int(ox1 - bw * 0.1))
            y1 = max(0, int(oy1 - bh * 0.1))
            x2 = min(orig_w, int(ox2 + bw * 0.1))
            y2 = min(orig_h, int(oy2 + bh * 0.1))
            if x2 <= x1 or y2 <= y1:
                continue
            # store box in original coords
            prop["box"] = [round(ox1, 1), round(oy1, 1), round(ox2, 1), round(oy2, 1)]
            valid_props.append(prop)
            valid_crops.append(target_original.crop((x1, y1, x2, y2)))

        sims = dino_similarities_batch(valid_crops, query_emb, dino_processor, dino_model, device)

        passed = 0
        best_sim = max(sims) if sims else 0.0
        for prop, sim in zip(valid_props, sims):
            if sim >= dino_threshold:
                all_detections.append({
                    "box":       prop["box"],
                    "owl_score": prop["score"],
                    "dino_sim":  round(sim, 4),
                })
                passed += 1

        print(f"[crop {qi}] {qpath.name} — {len(proposals)} proposals, "
              f"best_sim={best_sim:.3f}, {passed} passed ≥ {dino_threshold}")

    # Step 3 — final cross-crop NMS to merge overlapping detections
    if all_detections:
        boxes_t  = torch.tensor([d["box"] for d in all_detections])
        scores_t = torch.tensor([d["dino_sim"] for d in all_detections])
        keep     = torch.ops.torchvision.nms(boxes_t, scores_t, nms_threshold)
        detections = [all_detections[i] for i in keep]
    else:
        detections = []

    print(f"\n[results] {len(detections)} detection(s) after per-crop pipeline + NMS (sim ≥ {dino_threshold})")
    for i, det in enumerate(detections):
        print(f"  [{i}] owl={det['owl_score']}  dino={det['dino_sim']}  box={det['box']}")

    if show:
        visualize(target_original, detections, dino_threshold)


if __name__ == "__main__":
    main()
