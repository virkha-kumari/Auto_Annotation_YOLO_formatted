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
    dino_batch_size, yoloe_batch_size, output_dir,
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
    script     = str(Path(__file__).parent / "scripts" / "auto_annotate.py")
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
        "--small-obj-thresh",   "0.01",
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

        # ── PAGE 1 ────────────────────────────────────────────────────────────
        with gr.Group(visible=True) as page1:
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
                padding_p1 = gr.Number(value=0.05, precision=3,
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
                dino_batch_size = gr.Slider(4, 64, value=16, step=4,
                    label="DINOv2 batch size",
                    info="Embedding batch size. Reduce if VRAM OOM during DINOv2 phase.")

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

        # Page navigation
        def to_page(p1_vis, p2_vis, p3_vis):
            return gr.update(visible=p1_vis), gr.update(visible=p2_vis), gr.update(visible=p3_vis)

        next_to_p2.click(
            lambda classes_txt, images_dir, labels_dir: (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                classes_txt,
                images_dir,
                labels_dir,
                "",   # crops_dir_state — empty = derive from images_dir_p1
            ),
            inputs=[classes_txt_p1, images_dir_p1, labels_dir_p1],
            outputs=[page1, page2, page3, classes_txt_p2, source_images_dir_p2, source_labels_dir_p2, crops_dir_state],
        ).then(
            update_pipeline_classes, classes_txt_p2, class_dropdown_p2
        )

        back_to_p1.click(
            lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
            outputs=[page1, page2, page3],
        )

        next_to_p3.click(
            lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
            outputs=[page1, page2, page3],
        )

        back_to_p2.click(
            lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
            outputs=[page1, page2, page3],
        )

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

        def do_skip(skip_dir, classes_txt, images_dir, labels_dir):
            if not skip_dir or not Path(skip_dir).is_dir():
                return (
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    "", classes_txt, images_dir, labels_dir,
                )
            return (
                gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                skip_dir, classes_txt, images_dir, labels_dir,
            )

        skip_to_p2_btn.click(
            do_skip,
            inputs=[skip_crops_dir, classes_txt_p1, images_dir_p1, labels_dir_p1],
            outputs=[page1, page2, page3, crops_dir_state, classes_txt_p2, source_images_dir_p2, source_labels_dir_p2],
        ).then(update_pipeline_classes, classes_txt_p2, class_dropdown_p2)

        # Pipeline
        run_pipeline_btn.click(
            run_pipeline,
            inputs=[
                images_dir_p1, crops_dir_state,
                classes_txt_p2, class_dropdown_p2,
                targets_dir_p2, source_images_dir_p2, source_labels_dir_p2,
                yoloe_conf, nms_iou, wbf_score,
                dino_thresh, result_panel3_thresh, final_containment_thresh,
                sam2_mask_padding, sam_score_min, sam_area_min,
                dino_batch_size, yoloe_batch_size, output_dir_p2,
            ],
            outputs=[pipeline_log, next_to_p3, result_images_state],
        )

        # Gallery
        def refresh_gallery(output_dir, classes_txt, selected_classes):
            return collect_results(output_dir, classes_txt, selected_classes)

        refresh_btn.click(
            refresh_gallery,
            inputs=[output_dir_p2, classes_txt_p2, class_dropdown_p2],
            outputs=[result_gallery],
        )

        next_to_p3.click(
            refresh_gallery,
            inputs=[output_dir_p2, classes_txt_p2, class_dropdown_p2],
            outputs=[result_gallery],
        )

        # Download
        download_all_btn.click(
            download_yolo_labels,
            inputs=[output_dir_p2, save_folder_p3, classes_txt_p2, class_dropdown_p2],
            outputs=[download_status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Neon())
