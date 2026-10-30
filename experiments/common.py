"""Shared definitions: paths, label space, box geometry, image corruptions."""
import io, os, json
import numpy as np
from PIL import Image, ImageFilter

ROOT = os.environ.get("CXR_ROOT", "/mnt/hdd/cxr_vlm")
DATA, PREP, RUNS = f"{ROOT}/data", f"{ROOT}/prepared", f"{ROOT}/runs"
MAX_SIDE = 1024                       # every prepared image is resized so its longest side is at most this

# ---- shared finding space F ------------------------------------------------------------------------------
# Claims are labelled by CheXbert (14 classes); evidence comes from each dataset's box classes.
FINDINGS = ["atelectasis", "cardiomegaly", "consolidation", "effusion", "pneumothorax",
            "nodule_mass", "fracture", "pleural_thickening", "opacity"]
PROMPT_NAME = {"atelectasis": "atelectasis", "cardiomegaly": "cardiomegaly", "consolidation": "consolidation",
               "effusion": "pleural effusion", "pneumothorax": "pneumothorax", "nodule_mass": "lung nodule or mass",
               "fracture": "rib fracture", "pleural_thickening": "pleural thickening", "opacity": "lung opacity"}
CHEXBERT = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
            "Support Devices", "No Finding"]
CHEXBERT_TO_F = {"Cardiomegaly": "cardiomegaly", "Lung Opacity": "opacity", "Lung Lesion": "nodule_mass",
                 "Consolidation": "consolidation", "Pneumonia": "consolidation", "Atelectasis": "atelectasis",
                 "Pneumothorax": "pneumothorax", "Pleural Effusion": "effusion", "Pleural Other": "pleural_thickening",
                 "Fracture": "fracture", "Edema": "edema", "Enlarged Cardiomediastinum": "enlarged_cardiomediastinum"}
UNVERIFIABLE = {"edema", "enlarged_cardiomediastinum"}   # claimed findings no dataset here has boxes for

CHESTXDET_TO_F = {"Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Consolidation": "consolidation",
                  "Effusion": "effusion", "Pneumothorax": "pneumothorax", "Nodule": "nodule_mass", "Mass": "nodule_mass",
                  "Diffuse Nodule": "nodule_mass", "Fracture": "fracture", "Pleural Thickening": "pleural_thickening",
                  "Calcification": None, "Emphysema": None, "Fibrosis": None}
# A generic "lung opacity" claim is supported by any parenchymal opacity box.
OPACITY_SOURCES = {"Atelectasis", "Consolidation", "Nodule", "Mass", "Diffuse Nodule", "Fibrosis", "Calcification",
                   "tb_lesion"}


# ---- geometry ----------------------------------------------------------------------------------------------
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def rotate_box(box, deg, w, h):
    """Bounding box of a box rotated by `deg` about the image centre (PIL rotate convention, counter-clockwise)."""
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t); cx, cy = w / 2, h / 2
    xs, ys = np.array([box[0], box[2], box[2], box[0]]) - cx, np.array([box[1], box[1], box[3], box[3]]) - cy
    rx, ry = cx + c * xs + s * ys, cy - s * xs + c * ys
    return [float(np.clip(rx.min(), 0, w)), float(np.clip(ry.min(), 0, h)),
            float(np.clip(rx.max(), 0, w)), float(np.clip(ry.max(), 0, h))]


# ---- corruptions (after MedMNIST-C), three severities each --------------------------------------------------
CORRUPTIONS = {"noise": [0.04, 0.08, 0.16], "blur": [1.0, 2.0, 4.0], "jpeg": [30, 15, 5],
               "contrast": [0.6, 0.4, 0.25], "rotate": [5, 10, 20]}


def corrupt(img, kind, sev, seed=0):
    """img: PIL RGB. Returns (PIL RGB, box_fn) where box_fn maps an evidence box to the corrupted frame."""
    if kind is None:
        return img, (lambda b: b)
    p = CORRUPTIONS[kind][sev - 1]
    x = np.asarray(img.convert("L"), dtype=np.float32) / 255
    w, h = img.size
    if kind == "noise":
        x = np.clip(x + np.random.default_rng(seed).normal(0, p, x.shape), 0, 1)
    elif kind == "contrast":
        x = np.clip(0.5 + (x - 0.5) * p, 0, 1)
    elif kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(p * max(w, h) / 512)), (lambda b: b)
    elif kind == "jpeg":
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=p); buf.seek(0)
        return Image.open(buf).convert("RGB"), (lambda b: b)
    elif kind == "rotate":
        return img.rotate(p, resample=Image.BILINEAR, fillcolor=(0, 0, 0)), (lambda b: rotate_box(b, p, w, h))
    return Image.fromarray((x * 255).round().astype(np.uint8)).convert("RGB"), (lambda b: b)


def _fill(img, b, others, blur):
    """Replace box b by the mirrored contralateral region, or by a heavy blur if that region holds a lesion too."""
    w = img.width
    m = [w - b[2], b[1], w - b[0], b[3]]
    src = img.transpose(Image.FLIP_LEFT_RIGHT) if not any(iou(m, o) > 0 for o in others) else blur
    box = tuple(int(round(v)) for v in b)
    if box[2] > box[0] and box[3] > box[1]:
        img.paste(src.crop(box), box)


def mask_evidence(img, record, mode, seed=0):
    """Evidence ablation. mode 'evmask': remove every expert-marked region; 'randmask': same fill applied to boxes
    of the same sizes at random positions that touch no expert region (control for the editing itself)."""
    boxes = []
    for bs in record.get("evidence", {}).values():
        for b in bs:
            if b not in boxes:
                boxes.append(b)
    out = img.copy()
    blur = img.filter(ImageFilter.GaussianBlur(0.08 * max(img.size)))
    if mode == "evmask":
        for b in boxes:
            _fill(out, b, boxes, blur)
        return out, boxes
    # control boxes go inside approximate lung zones (left / right of the mediastinum), like real lesions
    rng, placed = np.random.default_rng(seed), []
    w, h = img.size
    zones = [(0.06 * w, 0.42 * w), (0.58 * w, 0.94 * w)]
    y0, y1 = 0.10 * h, 0.82 * h
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        for t in range(400):
            lo, hi = zones[rng.integers(2)] if t < 300 else (0.04 * w, 0.96 * w)    # very wide boxes: whole chest
            if hi - lo < bw or y1 - y0 < bh:
                continue
            x, y = rng.uniform(lo, hi - bw), rng.uniform(y0, y1 - bh)
            c = [x, y, x + bw, y + bh]
            if not any(iou(c, o) > 0 for o in boxes + placed):
                placed.append(c); break
    for c in placed:
        _fill(out, c, boxes, blur)
    return out, placed


def load_manifest(site):
    return [json.loads(l) for l in open(f"{PREP}/{site}.jsonl")]


if __name__ == "__main__":      # self-check
    assert abs(iou([0, 0, 2, 2], [1, 1, 3, 3]) - 1 / 7) < 1e-9
    assert rotate_box([10, 10, 20, 20], 0, 100, 100) == [10, 10, 20, 20]
    b = rotate_box([40, 40, 60, 60], 90, 100, 100); assert np.allclose(b, [40, 40, 60, 60]), b
    im = Image.new("RGB", (64, 32), (128, 128, 128))
    for k in CORRUPTIONS:
        for s in (1, 2, 3):
            out, f = corrupt(im, k, s); assert out.size == im.size and len(f([1, 1, 5, 5])) == 4
    big = Image.fromarray(np.tile(np.arange(100, dtype=np.uint8), (100, 1))).convert("RGB")
    rec = {"evidence": {"x": [[10, 10, 30, 30]]}}
    ev, bx = mask_evidence(big, rec, "evmask")
    a = np.asarray(ev)[10:30, 10:30, 0]; assert (a == np.arange(89, 69, -1)[None, :]).all(), "mirror fill"
    rc, pl = mask_evidence(big, rec, "randmask", seed=1)
    assert len(pl) == 1 and iou(pl[0], [10, 10, 30, 30]) == 0
    print("common.py ok")
