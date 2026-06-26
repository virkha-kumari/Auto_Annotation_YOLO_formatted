"""
Extract cropped object images from a YOLO-annotated dataset,
then cluster with DINOv2+DBSCAN to select unique representatives.

Per-class pipeline (each class fully independent):
  Phase 1 — for each class: scan all strided frames, extract crops,
             phash dedup within class → write candidates to _tmp/cls{id}/
  Phase 2 — load DINOv2 once, then for each class:
             embed candidates → L2-normalize → DBSCAN (cosine) →
             medoid per cluster (largest first) + farthest-point outliers
             → copy up to --max-per-class-crops winners to output/cls{id}/
             → delete tmp

Usage:
    python scripts/extract_crops_labelled.py \
        --images "D:/path/to/images" \
        --labels "D:/path/to/labels" \
        --classes 0 2 17 \
        --output "D:/path/to/crops" \
        --stride 2 \
        --padding 0.05 \
        --min-hash-dist 4 \
        --dbscan-eps 0.15 \
        --dbscan-min-samples 2 \
        --max-per-class-crops 100

Each saved crop is named:
    <frame_stem>_cls<class_id>_<ann_idx>.jpg
"""

import argparse
import gc
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
JPEG_QUALITY = 92


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract YOLO crops per class, dedup, cluster with DINOv2+DBSCAN, save unique representatives."
    )
    p.add_argument("--images",               required=True, help="Folder containing frame images")
    p.add_argument("--labels",               required=True, help="Folder containing YOLO .txt label files")
    p.add_argument("--classes",              required=True, nargs="+", type=int,
                   help="Class IDs to extract (space-separated, e.g. 0 2 17)")
    p.add_argument("--output",               required=True, help="Folder to save selected crops into")
    p.add_argument("--stride",               type=int,   default=2,
                   help="Process every Nth image (default: 2)")
    p.add_argument("--padding",              type=float, default=0.0,
                   help="Fractional padding around each box, e.g. 0.05 = 5%% (default: 0)")
    p.add_argument("--min-hash-dist",        type=int,   default=6,
                   help="Perceptual hash distance threshold for within-class dedup (default: 6, 0=off)")
    p.add_argument("--dbscan-eps",           type=float, default=0.15,
                   help="DBSCAN epsilon in cosine distance space (default: 0.15, ignored if --auto-tune)")
    p.add_argument("--dbscan-min-samples",   type=int,   default=2,
                   help="DBSCAN min_samples (default: 2)")
    p.add_argument("--auto-tune",            action="store_true",
                   help="Auto-find optimal eps via k-nearest-neighbors (overrides --dbscan-eps)")
    p.add_argument("--auto-tune-percentile", type=int,   default=85,
                   help="Percentile for auto-tune eps (default: 85)")
    p.add_argument("--max-per-class-crops",  type=int,   default=100,
                   help="Hard cap on saved crops per class: cluster reps + outliers (default: 100)")
    p.add_argument("--batch-size",           type=int,   default=32,
                   help="DINOv2 embedding batch size (default: 32)")
    return p.parse_args()


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h, padding=0.0):
    pad_x = w * padding
    pad_y = h * padding
    x1 = max(0,     int((cx - w / 2 - pad_x) * img_w))
    y1 = max(0,     int((cy - h / 2 - pad_y) * img_h))
    x2 = min(img_w, int((cx + w / 2 + pad_x) * img_w))
    y2 = min(img_h, int((cy + h / 2 + pad_y) * img_h))
    return x1, y1, x2, y2


def auto_tune_eps(normed_embeddings, min_samples, percentile):
    nn = NearestNeighbors(n_neighbors=min_samples, metric='cosine')
    nn.fit(normed_embeddings)
    distances, _ = nn.kneighbors(normed_embeddings)
    return float(np.percentile(np.sort(distances[:, -1]), percentile))


def farthest_point_sample(vectors, k):
    if k <= 0 or len(vectors) == 0:
        return []
    k = min(k, len(vectors))
    seed = int(np.argmax(np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)))
    selected = [seed]
    dists = np.full(len(vectors), np.inf)
    for _ in range(k - 1):
        last = vectors[selected[-1]]
        d = np.linalg.norm(vectors - last, axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return selected


def cluster_and_select(normed, max_crops, dbscan_eps, dbscan_min_samples,
                       auto_tune, auto_tune_percentile):
    """
    Cluster normed (N, 768) unit vectors, select up to max_crops representatives.
    Returns list of selected indices into normed.
    """
    if auto_tune:
        eps = auto_tune_eps(normed, dbscan_min_samples, auto_tune_percentile)
        print(f"    auto-tuned eps={eps:.4f} (p{auto_tune_percentile})")
    else:
        eps = dbscan_eps

    print(f"    DBSCAN eps={eps:.4f} min_samples={dbscan_min_samples} ...")
    labels = DBSCAN(eps=eps, min_samples=dbscan_min_samples, metric="cosine").fit_predict(normed)

    cluster_ids = [l for l in set(labels) if l != -1]
    noise_mask  = labels == -1
    print(f"    {len(cluster_ids)} cluster(s), {int(noise_mask.sum())} noise point(s)")

    # medoid per cluster, largest clusters first
    sorted_clusters = sorted(cluster_ids, key=lambda c: int((labels == c).sum()), reverse=True)
    selected = []
    for cid in sorted_clusters:
        if len(selected) >= max_crops:
            break
        mask        = labels == cid
        vecs        = normed[mask]
        centroid    = vecs.mean(axis=0)
        global_idxs = np.where(mask)[0]
        medoid      = int(global_idxs[np.argmin(np.linalg.norm(vecs - centroid, axis=1))])
        selected.append(medoid)

    # fill remaining with diverse outliers via farthest-point sampling
    remaining = max_crops - len(selected)
    if remaining > 0 and noise_mask.sum() > 0:
        noise_idxs = np.where(noise_mask)[0]
        fps_local  = farthest_point_sample(normed[noise_mask], remaining)
        selected.extend(int(noise_idxs[i]) for i in fps_local)

    n_reps     = sum(1 for i in selected if not noise_mask[i])
    n_outliers = len(selected) - n_reps
    print(f"    {n_reps} cluster rep(s) + {n_outliers} outlier(s) = {len(selected)} selected")
    return selected


def embed_batch(paths, processor, model, device, batch_size, use_amp, vram_total):
    """
    Embed a list of image paths using already-loaded DINOv2 model.
    Returns (N, 768) float32 numpy array.
    """
    import torch

    def _load(p):
        img = cv2.imread(str(p))
        if img is None:
            return Image.open(p).convert("RGB")
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    embeddings = []
    with torch.no_grad(), ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0, len(paths), batch_size):
            if use_amp and torch.cuda.memory_allocated() > 0.80 * vram_total:
                torch.cuda.empty_cache()
                gc.collect()

            batch_paths = paths[i:i + batch_size]
            batch = list(pool.map(_load, batch_paths))

            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(**inputs)
            else:
                out = model(**inputs)

            cls_tokens = out.last_hidden_state[:, 0, :]
            result = cls_tokens.cpu().float()
            if use_amp:
                torch.cuda.synchronize()
            embeddings.append(result.numpy())

            del inputs, out, batch, result
            if use_amp:
                torch.cuda.empty_cache()

    return np.vstack(embeddings)


def phase1_extract_class(cls_id, work_items, tmp_cls_dir, padding, min_hash_dist):
    """
    Extract + phash-dedup all crops for one class from pre-parsed work_items.
    work_items: list of (img_path, annotations) where annotations = list of
                (ann_idx, cls_id_int, cx, cy, w, h) already filtered to ALL classes.
    Writes JPEGs to tmp_cls_dir. Returns list of filenames written.
    """

    def _process_frame(img_path, annotations):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            return []
        ih, iw = bgr.shape[:2]
        results = []
        for ann_idx, cid, cx, cy, w, h in annotations:
            if cid != cls_id:
                continue
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, iw, ih, padding)
            if x2 <= x1 or y2 <= y1:
                results.append(None)
                continue
            crop_bgr = bgr[y1:y2, x1:x2].copy()
            bits = None
            if min_hash_dist > 0:
                ph = imagehash.phash(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
                bits = np.array(ph.hash, dtype=np.uint8).ravel()
            name = f"{img_path.stem}_cls{cid}_{ann_idx}.jpg"
            results.append((name, bits, crop_bgr))
        return results

    hash_mat = np.empty((256, 64), dtype=np.uint8)
    n_hashes = 0
    skipped_zero = 0
    candidate_names = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_process_frame, img_path, anns) for img_path, anns in work_items]
        for future in tqdm(futures, desc=f"  cls{cls_id} extract", unit="img", leave=False):
            for item in future.result():
                if item is None:
                    skipped_zero += 1
                    continue
                name, bits, crop_bgr = item
                if bits is not None:
                    if n_hashes > 0:
                        dists = (hash_mat[:n_hashes] != bits).sum(axis=1)
                        if dists.min() <= min_hash_dist:
                            continue
                    if n_hashes >= len(hash_mat):
                        hash_mat = np.vstack([hash_mat, np.empty_like(hash_mat)])
                    hash_mat[n_hashes] = bits
                    n_hashes += 1
                cv2.imwrite(str(tmp_cls_dir / name), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                candidate_names.append(name)

    if skipped_zero:
        print(f"  cls{cls_id}: skipped {skipped_zero} zero-area box(es)")
    return candidate_names


def main():
    args = parse_args()

    images_dir    = Path(args.images)
    labels_dir    = Path(args.labels)
    output_dir    = Path(args.output)
    target_classes = sorted(set(args.classes))

    if not images_dir.is_dir():
        sys.exit(f"[error] Images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        sys.exit(f"[error] Labels folder not found: {labels_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for cls_id in target_classes:
        (output_dir / f"cls{cls_id}").mkdir(exist_ok=True)

    img_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not img_paths:
        sys.exit(f"[error] No images found in {images_dir}")

    strided = img_paths[::args.stride]
    print(f"[scan] {len(img_paths)} images → {len(strided)} after stride={args.stride}")

    # Pre-parse all labels once (cheap) — workers reuse this for every class
    print("[scan] Parsing label files...")
    work_items = []   # (img_path, [(ann_idx, cls_id, cx, cy, w, h), ...])
    target_set = set(target_classes)
    for img_path in tqdm(strided, desc="  parsing labels", unit="img"):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        annotations = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"[warn] Skipping malformed line in {label_path}: {line}")
                continue
            cid = int(parts[0])
            if cid in target_set:
                annotations.append((len(annotations), cid,
                                     float(parts[1]), float(parts[2]),
                                     float(parts[3]), float(parts[4])))
        if annotations:
            work_items.append((img_path, annotations))

    print(f"[scan] {len(work_items)} frame(s) with target-class annotations")

    tmp_dir = output_dir / "_tmp_candidates"
    tmp_dir.mkdir(exist_ok=True)

    # --- Phase 1: per-class extract + dedup ---
    print("\n[phase1] Extracting and deduplicating crops per class...")
    class_candidates = {}   # cls_id -> list of filenames in tmp/cls{id}/
    for cls_id in target_classes:
        tmp_cls_dir = tmp_dir / f"cls{cls_id}"
        tmp_cls_dir.mkdir(exist_ok=True)
        print(f"\n  [cls{cls_id}] extracting...")
        names = phase1_extract_class(cls_id, work_items, tmp_cls_dir,
                                     args.padding, args.min_hash_dist)
        class_candidates[cls_id] = names
        print(f"  [cls{cls_id}] {len(names)} candidate(s) after dedup")

    total_candidates = sum(len(v) for v in class_candidates.values())
    if total_candidates == 0:
        shutil.rmtree(tmp_dir)
        print("[warn] No crops collected — check --classes match label files")
        return

    # --- Phase 2: load DINOv2 once, cluster per class ---
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    vram_total = torch.cuda.get_device_properties(0).total_memory if use_amp else 0

    print(f"\n[phase2] Loading DINOv2 on {device}...")
    try:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model     = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    except OSError:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
        model     = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).to(device).eval()

    total_saved = 0

    for cls_id in target_classes:
        names = class_candidates[cls_id]
        tmp_cls_dir = tmp_dir / f"cls{cls_id}"
        out_cls_dir = output_dir / f"cls{cls_id}"

        print(f"\n[cls{cls_id}] {len(names)} candidate(s)")

        if not names:
            print(f"[cls{cls_id}] no candidates — skipping")
            continue

        # trivial: at or under cap — move all, skip clustering
        if len(names) <= args.max_per_class_crops:
            print(f"[cls{cls_id}] ≤ max-per-class-crops ({args.max_per_class_crops}), saving all")
            for name in names:
                shutil.move(str(tmp_cls_dir / name), str(out_cls_dir / name))
            total_saved += len(names)
            continue

        # embed
        print(f"[cls{cls_id}] embedding {len(names)} crop(s)...")
        tmp_paths  = [tmp_cls_dir / name for name in names]
        embeddings = embed_batch(tmp_paths, processor, model, device,
                                 args.batch_size, use_amp, vram_total)

        if len(embeddings) < 2:
            shutil.move(str(tmp_paths[0]), str(out_cls_dir / names[0]))
            total_saved += 1
            continue

        # cluster + select
        normed = normalize(embeddings, norm='l2')
        del embeddings
        gc.collect()

        selected = cluster_and_select(
            normed, args.max_per_class_crops,
            args.dbscan_eps, args.dbscan_min_samples,
            args.auto_tune, args.auto_tune_percentile
        )
        del normed
        gc.collect()

        for idx in selected:
            shutil.copy2(str(tmp_paths[idx]), str(out_cls_dir / names[idx]))
        total_saved += len(selected)
        print(f"[cls{cls_id}] saved {len(selected)} crop(s)")

    # cleanup
    del model
    if use_amp:
        torch.cuda.empty_cache()
    gc.collect()

    shutil.rmtree(tmp_dir)
    print(f"\n[done] {total_saved} total crop(s) saved to {output_dir}")


if __name__ == "__main__":
    main()
