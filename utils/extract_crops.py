"""
Extract cropped object images from a YOLO-annotated dataset,
then cluster with DINOv2+DBSCAN to select unique representatives.

Phase 1 — scan up to --max-images images (after stride), write all
           candidate crops to a temp dir on disk (perceptual-hash dedup).
           No RAM accumulation of image data.
Phase 2 — DINOv2-base embed (batched, from disk) → L2-normalize →
           DBSCAN (cosine, eps~0.15) → one medoid per cluster (largest first, capped by
           --max-output-crops) + farthest-point outliers filling remaining
           slots → copy winners to --output, delete temp dir.

Usage:
    python utils/extract_crops.py \
        --images "D:/path/to/images" \
        --labels "D:/path/to/labels" \
        --classes 0 2 17 \
        --output "D:/path/to/crops" \
        --max-images 200 \
        --stride 5 \
        --padding 0.05 \
        --min-hash-dist 8 \
        --dbscan-eps 1.5 \
        --dbscan-min-samples 2 \
        --max-outliers 15 \
        --max-output-crops 20

Each saved crop is named:
    <frame_stem>_cls<class_id>_<ann_idx>.png
"""

import argparse
import shutil
import sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract YOLO-annotated crops, cluster with DINOv2+DBSCAN, save unique representatives."
    )
    p.add_argument("--images",             required=True,  help="Folder containing frame images")
    p.add_argument("--labels",             required=True,  help="Folder containing YOLO .txt label files")
    p.add_argument("--classes",            required=True,  nargs="+", type=int,
                   help="Class IDs to extract (space-separated, e.g. 0 2 17)")
    p.add_argument("--output",             required=True,  help="Folder to save selected crops into")
    p.add_argument("--max-images",         type=int, default=None,
                   help="Scan at most this many images after stride (default: no limit)")
    p.add_argument("--stride",             type=int, default=1,
                   help="Process every Nth image (default: 1)")
    p.add_argument("--min-hash-dist",      type=int, default=20,
                   help="Perceptual hash distance threshold — skip near-duplicate crops (default: 20, 0=off)")
    p.add_argument("--padding",            type=float, default=0.0,
                   help="Fractional padding around each box, e.g. 0.05 = 5%% (default: 0)")
    p.add_argument("--dbscan-eps",         type=float, default=0.25,
                   help="DBSCAN epsilon in cosine distance space (default: 0.25, ignored if --auto-tune)")
    p.add_argument("--dbscan-min-samples", type=int,   default=2,
                   help="DBSCAN min_samples (default: 2)")
    p.add_argument("--auto-tune",          action="store_true",
                   help="Auto-find optimal eps via k-nearest-neighbors (overrides --dbscan-eps)")
    p.add_argument("--auto-tune-percentile", type=int, default=90,
                   help="Percentile for auto-tune eps (90=tight, 95=balanced, 98=loose, default: 90)")
    p.add_argument("--max-outliers",       type=int,   default=25,
                   help="Max noise-point crops to keep via farthest-point sampling (default: 25)")
    p.add_argument("--max-output-crops",   type=int,   default=25,
                   help="Hard cap on total saved crops: clusters + outliers (default: 25)")
    return p.parse_args()


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h, padding=0.0):
    """Convert YOLO normalised box to absolute pixel xyxy with optional padding."""
    pad_x = w * padding
    pad_y = h * padding
    x1 = (cx - w / 2 - pad_x) * img_w
    y1 = (cy - h / 2 - pad_y) * img_h
    x2 = (cx + w / 2 + pad_x) * img_w
    y2 = (cy + h / 2 + pad_y) * img_h
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w, int(x2))
    y2 = min(img_h, int(y2))
    return x1, y1, x2, y2


def auto_tune_eps(normed_embeddings, min_samples, percentile):
    """Find optimal DBSCAN eps using k-nearest neighbors on L2-normed embeddings."""
    nn = NearestNeighbors(n_neighbors=min_samples, metric='cosine')
    nn.fit(normed_embeddings)
    distances, _ = nn.kneighbors(normed_embeddings)
    k_distances = np.sort(distances[:, -1])
    optimal_eps = float(np.percentile(k_distances, percentile))
    return optimal_eps


def embed_from_disk(paths, batch_size=16):
    """
    Embed a list of image paths with DINOv2-base, in batches.
    Returns numpy array of shape (N, 768).
    Deletes model + empties CUDA cache before returning.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] Loading DINOv2 on {device} — embedding {len(paths)} crop(s) (batch={batch_size})...")

    try:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    except OSError:
        print("[embed] Network unavailable — loading DINOv2 from local cache...")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
        model = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).to(device).eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            out = model(**inputs)
            cls_tokens = out.last_hidden_state[:, 0, :]  # (B, 768)
            embeddings.append(cls_tokens.cpu().float().numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[embed] Done.")
    return np.vstack(embeddings)  # (N, 768)


def farthest_point_sample(vectors, k):
    """
    Greedy farthest-point sampling in euclidean space.
    Starts from the point most distant from the mean (best spread seed).
    vectors: numpy (N, D)
    k: how many points to pick
    Returns list of indices (length min(k, N)).
    """
    if k <= 0 or len(vectors) == 0:
        return []
    k = min(k, len(vectors))
    # seed: most isolated point from the mean
    seed = int(np.argmax(np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)))
    selected = [seed]
    dists = np.full(len(vectors), np.inf)
    for _ in range(k - 1):
        last = vectors[selected[-1]]
        d = np.linalg.norm(vectors - last, axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return selected


def main():
    args = parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)
    target_classes = set(args.classes)

    # clamp max_outliers so --max-outliers 50 --max-output-crops 20 doesn't break logic
    max_outliers = min(args.max_outliers, args.max_output_crops)

    if not images_dir.is_dir():
        sys.exit(f"[error] Images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        sys.exit(f"[error] Labels folder not found: {labels_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not img_paths:
        sys.exit(f"[error] No images found in {images_dir}")

    strided = img_paths[::args.stride]
    if args.max_images:
        strided = strided[:args.max_images]

    # temp dir lives inside output — same filesystem, fast copy/move
    tmp_dir = output_dir / "_tmp_candidates"
    tmp_dir.mkdir(exist_ok=True)

    seen_hashes = []
    skipped_zero = 0
    candidate_names = []  # ordered list of filenames written to tmp_dir

    print(f"[phase1] Scanning {len(strided)} image(s)...")

    for img_path in strided:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        annotations = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id in target_classes:
                annotations.append((cls_id, float(parts[1]), float(parts[2]),
                                     float(parts[3]), float(parts[4])))
        if not annotations:
            continue

        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size

        for ann_idx, (cls_id, cx, cy, w, h) in enumerate(annotations):
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, iw, ih, args.padding)
            if x2 <= x1 or y2 <= y1:
                skipped_zero += 1
                continue

            crop = img.crop((x1, y1, x2, y2))

            if args.min_hash_dist > 0:
                ph = imagehash.phash(crop)
                if any(ph - s <= args.min_hash_dist for s in seen_hashes):
                    continue
                seen_hashes.append(ph)

            name = f"{img_path.stem}_cls{cls_id}_{ann_idx}.png"
            crop.save(tmp_dir / name)
            candidate_names.append(name)

    print(f"[phase1] {len(candidate_names)} candidate crop(s) after hash filter")
    if skipped_zero:
        print(f"[warn]   Skipped {skipped_zero} zero-area box(es)")

    if not candidate_names:
        shutil.rmtree(tmp_dir)
        print("[warn] No crops collected — check --classes match label files")
        return

    # trivial: fewer candidates than cap — move all directly, skip clustering
    if len(candidate_names) <= args.max_output_crops:
        print(f"[info] {len(candidate_names)} candidates ≤ --max-output-crops ({args.max_output_crops}), saving all directly")
        for name in candidate_names:
            shutil.move(str(tmp_dir / name), str(output_dir / name))
        shutil.rmtree(tmp_dir)
        print(f"[done] Saved {len(candidate_names)} crop(s) to {output_dir}")
        return

    # --- Phase 2a: embed from disk ---
    tmp_paths = [tmp_dir / name for name in candidate_names]
    embeddings = embed_from_disk(tmp_paths)  # (N, 768)

    # guard: need at least 2 samples to cluster
    if len(embeddings) < 2:
        shutil.move(str(tmp_paths[0]), str(output_dir / candidate_names[0]))
        shutil.rmtree(tmp_dir)
        print(f"[done] Saved 1 crop(s) to {output_dir}")
        return

    # --- Phase 2b: normalize → DBSCAN (cosine) → select ---
    normed = normalize(embeddings, norm='l2')  # (N, 768) unit vectors — cosine dist = euclidean on unit sphere

    if args.auto_tune:
        eps = auto_tune_eps(normed, args.dbscan_min_samples, args.auto_tune_percentile)
        print(f"[cluster] Auto-tuned eps={eps:.4f} (percentile={args.auto_tune_percentile})")
    else:
        eps = args.dbscan_eps

    print(f"[cluster] Running DBSCAN (eps={eps:.4f}, min_samples={args.dbscan_min_samples}, metric=cosine)...")
    labels = DBSCAN(
        eps=eps,
        min_samples=args.dbscan_min_samples,
        metric="cosine"
    ).fit_predict(normed)

    cluster_ids = [l for l in set(labels) if l != -1]
    noise_mask  = labels == -1
    print(f"[cluster] {len(cluster_ids)} cluster(s), {int(noise_mask.sum())} noise point(s)")

    # one medoid per cluster, largest clusters first
    cluster_sizes   = {cid: int((labels == cid).sum()) for cid in cluster_ids}
    sorted_clusters = sorted(cluster_ids, key=lambda c: cluster_sizes[c], reverse=True)

    selected = []
    for cid in sorted_clusters:
        if len(selected) >= args.max_output_crops:
            break
        mask        = labels == cid
        vecs        = normed[mask]
        centroid    = vecs.mean(axis=0)
        global_idxs = np.where(mask)[0]
        medoid      = int(global_idxs[np.argmin(np.linalg.norm(vecs - centroid, axis=1))])
        selected.append(medoid)

    # fill remaining slots with diverse outliers
    remaining      = args.max_output_crops - len(selected)
    outlier_budget = min(max_outliers, remaining)

    if outlier_budget > 0 and noise_mask.sum() > 0:
        noise_idxs = np.where(noise_mask)[0]
        fps_local  = farthest_point_sample(normed[noise_mask], outlier_budget)
        selected.extend(int(noise_idxs[i]) for i in fps_local)

    n_cluster_reps = sum(1 for i in selected if not noise_mask[i])
    n_outliers     = len(selected) - n_cluster_reps
    print(f"[select] {n_cluster_reps} cluster rep(s) + {n_outliers} outlier(s) = {len(selected)} total")

    # copy winners to output, delete temp dir
    for idx in selected:
        shutil.copy2(str(tmp_paths[idx]), str(output_dir / candidate_names[idx]))

    shutil.rmtree(tmp_dir)
    print(f"[done] Saved {len(selected)} crop(s) to {output_dir}")


if __name__ == "__main__":
    main()
