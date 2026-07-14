"""
Auto-Annotation Gradio App — 3-page wizard.

Page 1: Dataset setup + crop extraction
Page 2: Pipeline args + run (YOLOe → SAM2 → DINOv2 → WBF)
Page 3: Results viewer + YOLO .txt download
"""

import json
import os
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import gradio as gr


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def browse_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    path = filedialog.askdirectory()
    root.destroy()
    return path or ""


def browse_file() -> str:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    root.destroy()
    return path or ""


def load_classes(classes_txt_path: str) -> list[str]:
    p = Path(classes_txt_path)
    if not p.exists():
        return []
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return lines


def class_choices(classes: list[str]) -> list[str]:
    return [f"{i}: {name}" for i, name in enumerate(classes)]


def parse_selected_ids(selected: list[str]) -> list[int]:
    return [int(s.split(":")[0]) for s in selected]


def run_subprocess(cmd: list[str], log_fn):
    """Run a subprocess, stream stdout+stderr line-by-line via log_fn callback."""
    log_fn(f"[cmd] {' '.join(cmd)}\n")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    for line in proc.stdout:
        log_fn(line)
    proc.wait()
    return proc.returncode


def crops_dir_for(images_dir: str) -> Path:
    return Path(f"cropped_dir_{Path(images_dir).name}")


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — crop extraction
# ─────────────────────────────────────────────────────────────────────────────

def autofill_labels(images_dir: str) -> str:
    return images_dir


def update_class_dropdown(classes_txt: str):
    classes = load_classes(classes_txt)
    choices = class_choices(classes)
    return gr.update(choices=choices, value=[], interactive=bool(choices))


def is_tqdm_line(line: str) -> bool:
    """True for tqdm progress-bar-only lines (carriage-return lines, no real content)."""
    # tqdm emits lines like "  0%|          | 0/66 [00:00<?, ?frame/s]"
    # keep lines that have [ at start or phase/saved/done/abort/error markers
    s = line.strip()
    if not s:
        return False
    has_bar = "|" in s and ("%" in s or "it/s" in s or "s/it" in s or "frame/s" in s or "img/s" in s or "batch" in s or "prop" in s)
    has_content = any(s.startswith(p) for p in ("[", "=", "✅", "❌", "⚠", " Done", " Script"))
    return has_bar and not has_content


def run_crop_extraction(
    images_dir, labels_dir, classes_txt, selected_classes,
    crop_mode,
    stride, padding, min_hash_dist,
    dbscan_eps, dbscan_min_samples, auto_tune, auto_tune_percentile, max_per_class,
    batch_size,
):
    if not images_dir or not Path(images_dir).is_dir():
        yield "❌ Images dir not found.", gr.update(interactive=False)
        return
    if not labels_dir or not Path(labels_dir).is_dir():
        yield "❌ Labels dir not found.", gr.update(interactive=False)
        return
    if not selected_classes:
        yield "❌ Select at least one class.", gr.update(interactive=False)
        return

    cls_ids = parse_selected_ids(selected_classes)
    out_dir = str(crops_dir_for(images_dir))

    if crop_mode == "Unique crops (clustered)":
        script = str(Path(__file__).parent / "scripts" / "extract_crops_labelled.py")
        cmd = [
            sys.executable, script,
            "--images", images_dir,
            "--labels", labels_dir,
            "--classes", *[str(c) for c in cls_ids],
            "--output", out_dir,
            "--stride", str(stride),
            "--padding", str(padding),
            "--min-hash-dist", str(min_hash_dist),
            "--dbscan-eps", str(dbscan_eps),
            "--dbscan-min-samples", str(dbscan_min_samples),
            "--max-per-class-crops", str(max_per_class),
            "--batch-size", str(batch_size),
        ]
        if auto_tune:
            cmd += ["--auto-tune", "--auto-tune-percentile", str(int(auto_tune_percentile))]
    else:
        script = str(Path(__file__).parent / "scripts" / "extract_crops_varied.py")
        cmd = [
            sys.executable, script,
            "--images", images_dir,
            "--labels", labels_dir,
            "--classes", *[str(c) for c in cls_ids],
            "--output", out_dir,
            "--stride", str(stride),
            "--padding", str(padding),
        ]

    full_log = f"[cmd] {' '.join(cmd)}\n"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    yield full_log, gr.update(interactive=False)
    for line in proc.stdout:
        if not is_tqdm_line(line):
            full_log += line
            yield full_log, gr.update(interactive=False)
    proc.wait()

    if proc.returncode == 0:
        full_log += f"\nDone. Crops saved to: {out_dir}\n"
        yield full_log, gr.update(interactive=True)
    else:
        full_log += f"\nScript exited with code {proc.returncode}\n"
        yield full_log, gr.update(interactive=False)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — pipeline
# ─────────────────────────────────────────────────────────────────────────────

def update_pipeline_classes(classes_txt: str):
    classes = load_classes(classes_txt)
    choices = class_choices(classes)
    return gr.update(choices=choices, value=[], interactive=bool(choices))


def run_pipeline(
    images_dir_p1, crops_dir_override,
    classes_txt_p2, selected_classes_p2,
    targets_dir, source_images_dir, source_labels_dir,
    yoloe_conf, nms_iou, wbf_score,
    dino_thresh, result_panel3_thresh, final_containment_thresh,
    sam2_mask_padding, sam_score_min, sam_area_min,
    dino_batch_size, yoloe_batch_size, small_obj_thresh, output_dir,
):
    if not targets_dir or not Path(targets_dir).is_dir():
        yield "❌ Targets dir not found.", gr.update(interactive=False), []
        return
    if not source_images_dir or not Path(source_images_dir).is_dir():
        yield "❌ Source images dir not found.", gr.update(interactive=False), []
        return
    if not source_labels_dir or not Path(source_labels_dir).is_dir():
        yield "❌ Source labels dir not found.", gr.update(interactive=False), []
        return
    if not selected_classes_p2:
        yield "❌ Select at least one class to annotate.", gr.update(interactive=False), []
        return

    cls_ids    = parse_selected_ids(selected_classes_p2)
    classes    = load_classes(classes_txt_p2)
    crops_base = Path(crops_dir_override) if crops_dir_override else crops_dir_for(images_dir_p1)
    script     = str(Path(__file__).parent / "scripts" / "yoloe_sam2_dinov2_module.py")
    out_dir    = str(Path(output_dir))

    full_log          = ""
    all_result_images = []

    # validate crops dirs — skip classes with missing dirs
    valid_cls_ids   = []
    valid_crops_dirs = []
    for cls_id in cls_ids:
        qdir = str(crops_base / f"cls{cls_id}")
        if not Path(qdir).is_dir():
            cls_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)
            full_log += f"⚠️  cls{cls_id} ({cls_name}): crops dir not found ({qdir}) — skipping\n"
            yield full_log, gr.update(interactive=False), []
        else:
            valid_cls_ids.append(cls_id)
            valid_crops_dirs.append(qdir)

    if not valid_cls_ids:
        full_log += "❌ No valid crops dirs found.\n"
        yield full_log, gr.update(interactive=True), []
        return

    cls_names_str = ", ".join(
        f"cls{c} ({classes[c] if c < len(classes) else c})" for c in valid_cls_ids
    )
    full_log += f"\n{'='*60}\nRunning pipeline for: {cls_names_str}\n"
    yield full_log, gr.update(interactive=False), []

    # single subprocess call for ALL classes together
    cmd = [
        sys.executable, script,
        "--queries-dirs",       *valid_crops_dirs,
        "--class-ids",          *[str(c) for c in valid_cls_ids],
        "--targets-dir",        targets_dir,
        "--source-images",      source_images_dir,
        "--labels",             source_labels_dir,
        "--output-dir",         out_dir,
        "--yoloe-conf",         str(yoloe_conf),
        "--nms-iou",            str(nms_iou),
        "--wbf-score",          str(wbf_score),
        "--dino-thresh",        str(dino_thresh),
        "--result-thresh",      str(result_panel3_thresh),
        "--containment-thresh", str(final_containment_thresh),
        "--sam2-mask-padding",  str(sam2_mask_padding),
        "--sam-score-min",      str(sam_score_min),
        "--sam-area-min",       str(sam_area_min),
        "--dino-batch-size",    str(dino_batch_size),
        "--yoloe-batch-size",   str(int(yoloe_batch_size)),
        "--small-obj-thresh",   str(small_obj_thresh),
    ]

    full_log += f"[cmd] {' '.join(cmd)}\n"
    yield full_log, gr.update(interactive=False), []

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    for line in proc.stdout:
        if not is_tqdm_line(line):
            full_log += line
            yield full_log, gr.update(interactive=False), []
    proc.wait()

    if proc.returncode != 0:
        full_log += f"\n❌ Pipeline failed (exit {proc.returncode})\n"
        yield full_log, gr.update(interactive=False), []
        return

    full_log += f"\n✅ Done — {cls_names_str}\n"

    # collect preview images from summary.json for gallery
    summary_path = Path(out_dir) / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for target_name, info in summary.items():
            preview = Path(info.get("preview_file", ""))
            if preview.exists():
                n = info.get("n_final_total", "?")
                all_result_images.append((str(preview),
                    f"{Path(target_name).stem} | {n} box(es)"))

    yield full_log, gr.update(interactive=True), all_result_images


# ─────────────────────────────────────────────────────────────────────────────
# Page 2b — SAM3 pipeline
# ─────────────────────────────────────────────────────────────────────────────

def update_sam3_classes(refs_labels_dir: str):
    """SAM3 path has no classes.txt dropdown wired to names — discover ids directly
    from label files and show them as plain ids (no class-name mapping needed here)."""
    labels_path = Path(refs_labels_dir)
    if not labels_path.is_dir():
        return gr.update(choices=[], value=[], interactive=False)
    ids = set()
    for label_file in labels_path.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    ids.add(int(parts[0]))
                except ValueError:
                    pass
    choices = [str(i) for i in sorted(ids)]
    return gr.update(choices=choices, value=[], interactive=bool(choices))


def run_sam3_pipeline(
    refs_dir, refs_labels_dir, selected_classes, targets_dir, output_dir,
    orientation, split_ratio, canvas_size,
    threshold, batch_size,
    max_refs_per_class, dinov2_batch_size, phash_max_dist,
    dup_iou, containment_thresh,
    fp32, ref_jpeg_quality,
):
    if not refs_dir or not Path(refs_dir).is_dir():
        yield "❌ Refs dir not found.", gr.update(interactive=False), []
        return
    if not refs_labels_dir or not Path(refs_labels_dir).is_dir():
        yield "❌ Refs labels dir not found.", gr.update(interactive=False), []
        return
    if not targets_dir or not Path(targets_dir).is_dir():
        yield "❌ Targets dir not found.", gr.update(interactive=False), []
        return

    script = str(Path(__file__).parent / "scripts" / "sam3_dinov2_module.py")
    class_ids = selected_classes if selected_classes else ["all"]

    cmd = [
        sys.executable, script,
        "--refs-dir", refs_dir,
        "--refs-labels", refs_labels_dir,
        "--class-ids", *class_ids,
        "--targets-dir", targets_dir,
        "--output-dir", str(output_dir),
        "--orientation", orientation,
        "--split-ratio", str(split_ratio),
        "--canvas-size", str(int(canvas_size)),
        "--threshold", str(threshold),
        "--batch-size", str(int(batch_size)),
        "--max-refs-per-class", str(int(max_refs_per_class)),
        "--dinov2-batch-size", str(int(dinov2_batch_size)),
        "--phash-max-dist", str(int(phash_max_dist)),
        "--dup-iou-thresh", str(dup_iou),
        "--containment-thresh", str(containment_thresh),
        "--ref-jpeg-quality", str(int(ref_jpeg_quality)),
    ]
    if fp32:
        cmd.append("--fp32")

    full_log = f"[cmd] {' '.join(cmd)}\n"
    yield full_log, gr.update(interactive=False), []

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    for line in proc.stdout:
        if not is_tqdm_line(line):
            full_log += line
            yield full_log, gr.update(interactive=False), []
    proc.wait()

    if proc.returncode != 0:
        full_log += f"\n❌ Pipeline failed (exit {proc.returncode})\n"
        yield full_log, gr.update(interactive=False), []
        return

    full_log += "\n✅ Done\n"

    all_result_images = []
    summary_path = Path(output_dir) / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for target_name, info in summary.items():
            preview = Path(info.get("preview_file", ""))
            if preview.exists():
                n = info.get("n_final_total", "?")
                all_result_images.append((str(preview), f"{Path(target_name).stem} | {n} box(es)"))

    yield full_log, gr.update(interactive=True), all_result_images


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — results + download
# ─────────────────────────────────────────────────────────────────────────────

# Store final boxes in app state so download can access them
# key: (cls_id, target_stem) → (boxes_px [[x1,y1,x2,y2],...], img_w, img_h)
def collect_results(output_dir, classes_txt_p2, selected_classes_p2):
    """Scan summary.json, return (preview_path, caption) tuples for gallery."""
    gallery      = []
    summary_path = Path(output_dir) / "summary.json"
    if not summary_path.exists():
        return gallery
    summary = json.loads(summary_path.read_text())
    for target_name, info in summary.items():
        preview = Path(info.get("preview_file", ""))
        if not preview.exists():
            continue
        n = info.get("n_final_total", "?")
        gallery.append((str(preview), f"{Path(target_name).stem} | {n} box(es)"))
    return gallery


def download_yolo_labels(
    output_dir, save_folder,
    classes_txt_p2, selected_classes_p2,
):
    """Copy all YOLO .txt files from summary.json paths → save_folder."""
    if not save_folder:
        return "❌ Enter a save folder path."

    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    summary_path = Path(output_dir) / "summary.json"
    if not summary_path.exists():
        return "❌ summary.json not found — run pipeline first."

    summary = json.loads(summary_path.read_text())
    copied  = 0
    missing = 0

    for target_name, info in summary.items():
        label_src = Path(info.get("label_file", ""))
        if not label_src.exists():
            missing += 1
            continue
        dest = save_path / label_src.name
        shutil.copy2(str(label_src), str(dest))
        copied += 1

    msg = f"✅ {copied} label file(s) saved to {save_path}"
    if missing:
        msg += f"  ({missing} missing)"
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="Auto-Annotation Pipeline") as demo:

        gr.Markdown("# 🏭 Auto-Annotation Pipeline")

        # ── LANDING PAGE ──────────────────────────────────────────────────────
        with gr.Group(visible=True) as landing_page:
            gr.Markdown("## Choose Pipeline")
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.Markdown(
                        "### YOLOe → SAM2 → DINOv2\n"
                        "Visual-prompt detection. Needs a crop-extraction step first.\n\n"
                        "✅ Confirmed working on large industrial objects + PPE."
                    )
                    choose_yoloe_btn = gr.Button("▶ Use YOLOe → SAM2 → DINOv2", variant="primary")
                with gr.Column(variant="panel"):
                    gr.Markdown(
                        "### SAM3 → DINOv2\n"
                        "Canvas-composite few-shot exemplar prompting, directly on labelled "
                        "reference images.\n\n"
                        "⚡ No crop-extraction step needed."
                    )
                    choose_sam3_btn = gr.Button("▶ Use SAM3 → DINOv2", variant="primary")

        # ── PAGE 1 ────────────────────────────────────────────────────────────
        with gr.Group(visible=False) as page1:
            gr.Markdown("## Step 1 — Dataset & Crop Extraction")

            with gr.Row():
                classes_txt_p1 = gr.Textbox(
                    label="classes.txt path",
                    placeholder="D:/project/classes.txt",
                    info="YOLO names file — one class per line. Required.",
                )
                classes_browse_btn_p1 = gr.Button("📂 Browse", scale=0)
                load_classes_btn = gr.Button("Load classes", variant="secondary", scale=0)

            class_dropdown_p1 = gr.Dropdown(
                label="Classes to extract",
                choices=[], multiselect=True, interactive=False,
                info="Select one or more classes to extract crops for.",
            )

            with gr.Row():
                images_dir_p1 = gr.Textbox(
                    label="Images folder",
                    placeholder="D:/project/images",
                    info="Folder containing raw frame images.",
                )
                images_browse_btn_p1 = gr.Button("📂", scale=0, min_width=40)
            with gr.Row():
                labels_dir_p1 = gr.Textbox(
                    label="Labels folder",
                    placeholder="Auto-filled from images folder",
                    info="Folder with YOLO .txt label files. Auto-filled from images folder — change if different.",
                )
                labels_browse_btn_p1 = gr.Button("📂", scale=0, min_width=40)

            images_dir_p1.change(autofill_labels, images_dir_p1, labels_dir_p1)

            crop_mode = gr.Radio(
                choices=["Unique crops (clustered)", "All crops (no clustering)"],
                value="Unique crops (clustered)",
                label="Crop extraction mode",
                info="Unique: DINOv2+DBSCAN to pick diverse reps (recommended for 50-100 seed crops). All: save every crop as-is (use if you manually annotated specific frames).",
            )

            with gr.Row():
                stride_p1 = gr.Slider(1, 10, value=2, step=1,
                    label="Stride",
                    info="Process every Nth image. Stride=2 halves scan time.")
                padding_p1 = gr.Number(value=0.0, precision=3,
                    label="BBox padding",
                    info="Fractional padding around each crop bbox. 0.05 = 5% context around the object.")
                min_hash_dist_p1 = gr.Slider(0, 20, value=6, step=1,
                    label="Min phash distance",
                    info="Perceptual hash dedup threshold. Higher = more aggressive dedup. 0 = off.")

            with gr.Group(visible=True) as unique_crop_opts:
                gr.Markdown("**Clustering options** *(Unique crops mode only)*")
                with gr.Row():
                    dbscan_eps_p1 = gr.Number(value=0.15, precision=3,
                        label="DBSCAN eps",
                        info="Cosine distance threshold for clustering. Lower = tighter clusters (more reps). Ignored if auto-tune is on.")
                    dbscan_min_samples_p1 = gr.Number(value=2, precision=0,
                        label="DBSCAN min samples",
                        info="Min points to form a cluster. Usually keep at 2.")
                    max_per_class_p1 = gr.Number(value=100, precision=0,
                        label="Max crops per class",
                        info="Hard cap on saved crops. Cluster reps + diverse outliers up to this limit.")
                with gr.Row():
                    auto_tune_p1 = gr.Checkbox(value=False,
                        label="Auto-tune DBSCAN eps",
                        info="Finds optimal eps via k-nearest-neighbors on the embeddings. Overrides eps number above.")
                    auto_tune_percentile_p1 = gr.Number(value=85, precision=0,
                        label="KNN percentile",
                        info="Percentile of k-NN distances used to set eps. 85 = slightly above median cluster tightness.")
                    batch_size_p1 = gr.Slider(8, 64, value=32, step=8,
                        label="DINOv2 batch size",
                        info="Embedding batch size. Reduce if VRAM OOM during clustering.")

            crop_mode.change(
                lambda m: gr.update(visible=(m == "Unique crops (clustered)")),
                crop_mode, unique_crop_opts,
            )

            gr.Markdown("---")
            gr.Markdown("### Already have cropped images?")
            with gr.Row():
                skip_crops_dir = gr.Textbox(
                    label="Existing crops folder",
                    placeholder="D:/my_crops  (must contain cls0/, cls1/, ... subfolders)",
                    info="If you already extracted crops, point here and skip Step 1 extraction entirely.",
                )
                skip_crops_browse_btn = gr.Button("📂", scale=0, min_width=40)
            skip_to_p2_btn = gr.Button("Skip extraction → Go to Pipeline", variant="secondary")

            gr.Markdown("---")
            run_crops_btn = gr.Button("▶ Extract Crops", variant="primary")
            crops_log = gr.Textbox(label="Log", lines=12, interactive=False, autoscroll=True)
            next_to_p2 = gr.Button("Next → Pipeline", variant="secondary", interactive=False)

        # ── PAGE 2 ────────────────────────────────────────────────────────────
        with gr.Group(visible=False) as page2:
            gr.Markdown("## Step 2 — Annotation Pipeline")

            with gr.Row():
                classes_txt_p2 = gr.Textbox(
                    label="classes.txt path",
                    placeholder="Auto-filled from Step 1",
                    info="Same classes.txt as Step 1.",
                )
                classes_browse_btn_p2 = gr.Button("📂", scale=0, min_width=40)
                load_classes_btn_p2 = gr.Button("Reload classes", variant="secondary", scale=0)

            class_dropdown_p2 = gr.Dropdown(
                label="Classes to annotate",
                choices=[], multiselect=True, interactive=False,
                info="Pipeline runs once per selected class, sequentially.",
            )

            with gr.Row():
                targets_dir_p2 = gr.Textbox(
                    label="Targets folder (unlabeled)",
                    placeholder="D:/project/images/train",
                    info="Folder of unlabeled images to auto-annotate.",
                )
                targets_browse_btn_p2 = gr.Button("📂", scale=0, min_width=40)
            with gr.Row():
                source_images_dir_p2 = gr.Textbox(
                    label="Source images folder (labelled seed frames)",
                    placeholder="Auto-filled from Step 1",
                    info="Folder containing the labelled source images used to extract seed crops. Auto-filled from Step 1.",
                )
                source_images_browse_btn_p2 = gr.Button("📂", scale=0, min_width=40)
            with gr.Row():
                source_labels_dir_p2 = gr.Textbox(
                    label="Source labels folder",
                    placeholder="Auto-filled from Step 1",
                    info="YOLO .txt labels for the source images above. Auto-filled from Step 1 — change if different.",
                )
                source_labels_browse_btn_p2 = gr.Button("📂", scale=0, min_width=40)
            with gr.Row():
                output_dir_p2 = gr.Textbox(
                    label="Output folder",
                    value="output_results",
                    info="Results saved here under cls<id>/ subfolders.",
                )
                output_browse_btn_p2 = gr.Button("📂", scale=0, min_width=40)

            gr.Markdown("### YOLOe")
            with gr.Row():
                yoloe_conf = gr.Number(value=0.06, precision=3,
                    label="YOLOe confidence",
                    info="Min detection confidence for YOLOe proposals. Lower = more proposals (noisier). 0.06 confirmed working.")
                nms_iou = gr.Number(value=0.45, precision=3,
                    label="NMS IoU",
                    info="IoU threshold for non-max suppression inside YOLOe and WBF clustering. Higher = fewer merges.")
                yoloe_batch_size = gr.Slider(1, 32, value=8, step=1,
                    label="YOLOe target batch size",
                    info="Targets processed per YOLOe call (after VPE bake-in). Higher = faster but more VRAM. 8 is safe for 8GB.")

            gr.Markdown("### DINOv2 scoring")
            with gr.Row():
                dino_thresh = gr.Number(value=0.65, precision=3,
                    label="DINOv2 similarity threshold",
                    info="Min cosine similarity (masked-patch pooling) to keep a proposal. Below this = dropped before WBF.")
                dino_batch_size = gr.Slider(4, 64, value=32, step=4,
                    label="DINOv2 batch size",
                    info="Embedding batch size. Reduce if VRAM OOM during DINOv2 phase.")
                small_obj_thresh = gr.Number(value=0.02, precision=4,
                    label="Small object threshold",
                    info="Classes with p90 bbox area (w×h normalised) below this use CLS-token embedding instead of masked-patch pooling. Raise for tiny classes like gloves.")

            gr.Markdown("### WBF + filtering")
            with gr.Row():
                wbf_score = gr.Number(value=0.10, precision=3,
                    label="WBF min score",
                    info="Min consolidated score (0.3*yoloe + 0.7*dino, fused by WBF) to keep a box. Low value keeps all candidates visible in Panel 2.")
                result_panel3_thresh = gr.Number(value=0.5, precision=3,
                    label="Result display threshold (Panel 3)",
                    info="WBF boxes above this score shown in Panel 3 (cyan). Pre-containment result gate.")
                final_containment_thresh = gr.Number(value=0.7, precision=3,
                    label="Containment filter threshold (Panel 4)",
                    info="If intersection/min_area > this, lower-scoring nested box is dropped. Panel 4 shows final result.")

            gr.Markdown("### SAM2")
            with gr.Row():
                sam2_mask_padding = gr.Number(value=0.05, precision=3,
                    label="SAM2 bbox padding",
                    info="Fractional padding added to bbox before prompting SAM2. Helps SAM2 see object edges.")
                sam_score_min = gr.Number(value=0.50, precision=3,
                    label="SAM2 min score",
                    info="Min SAM2 mask quality score to accept a reference crop mask. Low scores = poor segmentation.")
                sam_area_min = gr.Number(value=0.10, precision=3,
                    label="SAM2 min area ratio",
                    info="Min ratio of mask pixels / bbox area. Filters masks that barely cover the object.")

            run_pipeline_btn = gr.Button("▶ Run Pipeline", variant="primary")
            pipeline_log = gr.Textbox(label="Log", lines=15, interactive=False, autoscroll=True)

            with gr.Row():
                back_to_p1 = gr.Button("← Back", variant="secondary")
                next_to_p3 = gr.Button("Next → Results", variant="secondary", interactive=False)

        # ── PAGE 2b — SAM3 ───────────────────────────────────────────────────
        with gr.Group(visible=False) as page2b:
            gr.Markdown("## SAM3 → DINOv2 Pipeline")

            with gr.Row():
                refs_dir_p2b = gr.Textbox(
                    label="Reference images folder",
                    placeholder="D:/project/labelled_ref_images",
                    info="Full labelled reference images (not crops) — SAM3 needs the whole frame to build the canvas.",
                )
                refs_dir_browse_btn_p2b = gr.Button("📂", scale=0, min_width=40)
            with gr.Row():
                refs_labels_dir_p2b = gr.Textbox(
                    label="Reference labels folder",
                    placeholder="Auto-filled from refs folder",
                    info="YOLO .txt labels, same stem as reference images.",
                )
                refs_labels_browse_btn_p2b = gr.Button("📂", scale=0, min_width=40)

            refs_dir_p2b.change(autofill_labels, refs_dir_p2b, refs_labels_dir_p2b)

            with gr.Row():
                class_dropdown_p2b = gr.Dropdown(
                    label="Classes to annotate",
                    choices=[], multiselect=True, interactive=False,
                    info="Leave empty to auto-discover + use every class id found in the labels.",
                )
                load_classes_btn_p2b = gr.Button("Discover classes", variant="secondary", scale=0)

            with gr.Row():
                targets_dir_p2b = gr.Textbox(
                    label="Targets folder (unlabeled)",
                    placeholder="D:/project/images/train",
                    info="Folder of unlabeled images to auto-annotate.",
                )
                targets_browse_btn_p2b = gr.Button("📂", scale=0, min_width=40)
            with gr.Row():
                output_dir_p2b = gr.Textbox(
                    label="Output folder",
                    value="output_sam3_dinov2",
                    info="Results saved here: previews/, labels/, temp_refs/, ref_crops_temp_embed/",
                )
                output_browse_btn_p2b = gr.Button("📂", scale=0, min_width=40)

            gr.Markdown("### Canvas-composite exemplar")
            with gr.Row():
                orientation_p2b = gr.Radio(choices=["vertical", "horizontal"], value="vertical",
                    label="Canvas orientation",
                    info="How ref and target halves are stacked on the shared canvas.")
                split_ratio_p2b = gr.Number(value=0.5, precision=2,
                    label="Split ratio",
                    info="Fraction of canvas given to the ref half. 0.5 = even split.")
                canvas_size_p2b = gr.Slider(504, 1512, value=1008, step=126,
                    label="Canvas size (px)",
                    info="Square canvas side length SAM3 processes.")

            gr.Markdown("### SAM3")
            with gr.Row():
                threshold_p2b = gr.Number(value=0.6, precision=3,
                    label="SAM3 score threshold",
                    info="post_process_object_detection score gate.")
                batch_size_p2b = gr.Slider(1, 16, value=8, step=1,
                    label="Target batch size",
                    info="Targets per SAM3 forward pass (same ref group). Lower if OOM on 8GB VRAM.")
                fp32_p2b = gr.Checkbox(value=False,
                    label="Force fp32",
                    info="Default is bf16 on CUDA. Check to disable.")

            gr.Markdown("### Diverse ref selection (DINOv2)")
            with gr.Row():
                max_refs_p2b = gr.Number(value=5, precision=0,
                    label="Max refs per class",
                    info="Diverse refs kept via DINOv2 + farthest-point sampling. 0 = use all refs.")
                dinov2_batch_size_p2b = gr.Slider(4, 64, value=32, step=4,
                    label="DINOv2 batch size",
                    info="Used for both ref embedding and target box scoring.")
                phash_max_dist_p2b = gr.Slider(0, 20, value=0, step=1,
                    label="Phash max dist",
                    info="Near-duplicate ref crop dedup before DINOv2 embedding. 0 = disabled.")
                ref_jpeg_quality_p2b = gr.Slider(50, 100, value=80, step=5,
                    label="Ref crop JPEG quality",
                    info="Quality for saved crops in output_dir/temp_refs/.")

            gr.Markdown("### Duplicate + containment filter")
            with gr.Row():
                dup_iou_p2b = gr.Number(value=0.85, precision=3,
                    label="Duplicate IoU",
                    info="Above this, two SAM3 proposals (same class+target) are merged: highest-score box's coords kept, scores averaged.")
                containment_thresh_p2b = gr.Number(value=0.85, precision=3,
                    label="Containment threshold",
                    info="Fully-inside-another-box ratio above which the lower-score box is dropped.")

            run_sam3_btn = gr.Button("▶ Run SAM3 Pipeline", variant="primary")
            sam3_log = gr.Textbox(label="Log", lines=15, interactive=False, autoscroll=True)

            with gr.Row():
                back_to_landing_from_2b = gr.Button("← Back", variant="secondary")
                next_to_p3_from_2b = gr.Button("Next → Results", variant="secondary", interactive=False)

        # ── PAGE 3 ────────────────────────────────────────────────────────────
        with gr.Group(visible=False) as page3:
            gr.Markdown("## Step 3 — Results & Download")
            gr.Markdown(
                "Panel 4 (orange) = final result after WBF + containment filter. "
                "Select images to download their YOLO labels."
            )

            refresh_btn = gr.Button("🔄 Refresh gallery", variant="secondary")
            result_gallery = gr.Gallery(
                label="Result images (Panel 4 = final boxes)",
                columns=3, height=600, object_fit="contain",
                show_label=True,
            )

            gr.Markdown("### Download YOLO labels")
            with gr.Row():
                save_folder_p3 = gr.Textbox(
                    label="Save labels to folder",
                    placeholder="D:/project/auto_labels",
                    info="Folder where .txt YOLO label files will be saved.",
                )
                save_browse_btn_p3 = gr.Button("📂", scale=0, min_width=40)
            download_all_btn = gr.Button("⬇ Download ALL labels", variant="primary")
            download_status  = gr.Textbox(label="Download status", interactive=False)

            back_to_p2 = gr.Button("← Back", variant="secondary")

        # ── WIRING ────────────────────────────────────────────────────────────

        crops_dir_state    = gr.State("")
        result_images_state = gr.State([])

        def show_only(which: str):
            """which in {landing, page1, page2, page2b, page3} — all 5 pages toggled explicitly."""
            return (
                gr.update(visible=which == "landing"),
                gr.update(visible=which == "page1"),
                gr.update(visible=which == "page2"),
                gr.update(visible=which == "page2b"),
                gr.update(visible=which == "page3"),
            )

        PAGE_OUTPUTS = [landing_page, page1, page2, page2b, page3]

        # Landing page choices
        choose_yoloe_btn.click(lambda: show_only("page1"), outputs=PAGE_OUTPUTS)
        choose_sam3_btn.click(lambda: show_only("page2b"), outputs=PAGE_OUTPUTS)

        next_to_p2.click(
            lambda classes_txt, images_dir, labels_dir: (
                *show_only("page2"),
                classes_txt,
                images_dir,
                labels_dir,
                "",   # crops_dir_state — empty = derive from images_dir_p1
            ),
            inputs=[classes_txt_p1, images_dir_p1, labels_dir_p1],
            outputs=[*PAGE_OUTPUTS, classes_txt_p2, source_images_dir_p2, source_labels_dir_p2, crops_dir_state],
        ).then(
            update_pipeline_classes, classes_txt_p2, class_dropdown_p2
        )

        back_to_p1.click(lambda: show_only("page1"), outputs=PAGE_OUTPUTS)
        next_to_p3.click(lambda: show_only("page3"), outputs=PAGE_OUTPUTS)
        back_to_p2.click(lambda: show_only("page2"), outputs=PAGE_OUTPUTS)

        back_to_landing_from_2b.click(lambda: show_only("landing"), outputs=PAGE_OUTPUTS)
        next_to_p3_from_2b.click(lambda: show_only("page3"), outputs=PAGE_OUTPUTS)

        # Browse buttons (tkinter folder/file dialogs)
        classes_browse_btn_p1.click(browse_file,   outputs=classes_txt_p1)
        images_browse_btn_p1.click( browse_folder, outputs=images_dir_p1)
        labels_browse_btn_p1.click( browse_folder, outputs=labels_dir_p1)
        classes_browse_btn_p2.click(browse_file,   outputs=classes_txt_p2)
        targets_browse_btn_p2.click(browse_folder, outputs=targets_dir_p2)
        source_images_browse_btn_p2.click(browse_folder, outputs=source_images_dir_p2)
        source_labels_browse_btn_p2.click(browse_folder, outputs=source_labels_dir_p2)
        output_browse_btn_p2.click( browse_folder, outputs=output_dir_p2)
        save_browse_btn_p3.click(   browse_folder, outputs=save_folder_p3)
        refs_dir_browse_btn_p2b.click(browse_folder, outputs=refs_dir_p2b)
        refs_labels_browse_btn_p2b.click(browse_folder, outputs=refs_labels_dir_p2b)
        targets_browse_btn_p2b.click(browse_folder, outputs=targets_dir_p2b)
        output_browse_btn_p2b.click(browse_folder, outputs=output_dir_p2b)

        # Class loading
        load_classes_btn.click(update_class_dropdown, classes_txt_p1, class_dropdown_p1)
        load_classes_btn_p2.click(update_pipeline_classes, classes_txt_p2, class_dropdown_p2)

        # Crop extraction
        run_crops_btn.click(
            run_crop_extraction,
            inputs=[
                images_dir_p1, labels_dir_p1, classes_txt_p1, class_dropdown_p1,
                crop_mode,
                stride_p1, padding_p1, min_hash_dist_p1,
                dbscan_eps_p1, dbscan_min_samples_p1, auto_tune_p1, auto_tune_percentile_p1, max_per_class_p1,
                batch_size_p1,
            ],
            outputs=[crops_log, next_to_p2],
        )

        # Browse + skip wiring
        skip_crops_browse_btn.click(browse_folder, outputs=skip_crops_dir)

        crops_dir_state = gr.State("")
        active_output_dir = gr.State("")  # which output dir Page3 reads from (YOLOe or SAM3 run)

        def do_skip(skip_dir, classes_txt, images_dir, labels_dir):
            if not skip_dir or not Path(skip_dir).is_dir():
                return (*show_only("page1"), "", classes_txt, images_dir, labels_dir)
            return (*show_only("page2"), skip_dir, classes_txt, images_dir, labels_dir)

        skip_to_p2_btn.click(
            do_skip,
            inputs=[skip_crops_dir, classes_txt_p1, images_dir_p1, labels_dir_p1],
            outputs=[*PAGE_OUTPUTS, crops_dir_state, classes_txt_p2, source_images_dir_p2, source_labels_dir_p2],
        ).then(update_pipeline_classes, classes_txt_p2, class_dropdown_p2)

        # Pipeline (YOLOe)
        run_pipeline_btn.click(
            run_pipeline,
            inputs=[
                images_dir_p1, crops_dir_state,
                classes_txt_p2, class_dropdown_p2,
                targets_dir_p2, source_images_dir_p2, source_labels_dir_p2,
                yoloe_conf, nms_iou, wbf_score,
                dino_thresh, result_panel3_thresh, final_containment_thresh,
                sam2_mask_padding, sam_score_min, sam_area_min,
                dino_batch_size, yoloe_batch_size, small_obj_thresh, output_dir_p2,
            ],
            outputs=[pipeline_log, next_to_p3, result_images_state],
        )

        # Pipeline (SAM3)
        load_classes_btn_p2b.click(update_sam3_classes, refs_labels_dir_p2b, class_dropdown_p2b)

        run_sam3_btn.click(
            run_sam3_pipeline,
            inputs=[
                refs_dir_p2b, refs_labels_dir_p2b, class_dropdown_p2b, targets_dir_p2b, output_dir_p2b,
                orientation_p2b, split_ratio_p2b, canvas_size_p2b,
                threshold_p2b, batch_size_p2b,
                max_refs_p2b, dinov2_batch_size_p2b, phash_max_dist_p2b,
                dup_iou_p2b, containment_thresh_p2b,
                fp32_p2b, ref_jpeg_quality_p2b,
            ],
            outputs=[sam3_log, next_to_p3_from_2b, result_images_state],
        )

        # Gallery — active_output_dir tracks which run (YOLOe/SAM3) Page3 should read
        def refresh_gallery(output_dir, classes_txt, selected_classes):
            return collect_results(output_dir, classes_txt, selected_classes)

        refresh_btn.click(
            refresh_gallery,
            inputs=[active_output_dir, classes_txt_p2, class_dropdown_p2],
            outputs=[result_gallery],
        )

        next_to_p3.click(lambda d: d, inputs=output_dir_p2, outputs=active_output_dir).then(
            refresh_gallery,
            inputs=[active_output_dir, classes_txt_p2, class_dropdown_p2],
            outputs=[result_gallery],
        )

        next_to_p3_from_2b.click(lambda d: d, inputs=output_dir_p2b, outputs=active_output_dir).then(
            refresh_gallery,
            inputs=[active_output_dir, classes_txt_p2, class_dropdown_p2],
            outputs=[result_gallery],
        )

        # Download
        download_all_btn.click(
            download_yolo_labels,
            inputs=[active_output_dir, save_folder_p3, classes_txt_p2, class_dropdown_p2],
            outputs=[download_status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Neon())
