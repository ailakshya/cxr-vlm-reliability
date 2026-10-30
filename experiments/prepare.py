"""Turn the raw datasets into one manifest per site: $CXR_ROOT/prepared/<site>.jsonl plus resized PNGs.

Each line: id, path, w, h, labels{...}, has_boxes, evidence{finding: [[x1,y1,x2,y2] px]}, verifiable[...],
report (reference text, if any), meta{...}. Images are resized so the longest side is <= MAX_SIDE.

Sites
  chestxdet      ChestX-Det official test split (553), boxes for 13 classes, 3 radiologists (NIH images)
  chestxdet_cal  500 images from the ChestX-Det train split, used only to calibrate conformal thresholds
  shenzhen       662 Shenzhen (China) images, TB label, lesion polygons from two readers (Data 2022 annotations)
  montgomery     138 Montgomery County (USA) images, TB label
  iuxray         Indiana (USA) frontal images with reference reports
  india_mendeley 799 images from KGMU Lucknow, UPUMS Saifai, GMC Kota (COVID / other abnormal / normal)
  india_nitrd    NITRD New Delhi TB Dataset A/B (TB / non-TB, two scanners)

Usage: python prepare.py [site ...]
"""
import glob, io, json, os, random, re, sys, collections
import numpy as np
from PIL import Image
from common import DATA, PREP, MAX_SIDE, FINDINGS, CHESTXDET_TO_F, OPACITY_SOURCES

Image.MAX_IMAGE_PIXELS = None


def save(site, rid, img):
    img = img.convert("RGB")
    s = min(1.0, MAX_SIDE / max(img.size))
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    d = f"{PREP}/{site}/img"; os.makedirs(d, exist_ok=True)
    p = f"{d}/{rid}.png"
    if not os.path.exists(p):
        img.save(p)
    return p, img.size, s


def write(site, recs):
    os.makedirs(PREP, exist_ok=True)
    with open(f"{PREP}/{site}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    ev = collections.Counter(k for r in recs for k, v in r.get("evidence", {}).items() if v)
    print(f"{site}: {len(recs)} images; images with evidence per finding: {dict(ev)}", flush=True)


def add(ev, f, box):
    ev.setdefault(f, []).append([round(float(v), 1) for v in box])


# ---- ChestX-Det ---------------------------------------------------------------------------------------------
def chestxdet():
    import pyarrow.parquet as pq
    d = f"{DATA}/chestxdet"
    imgs = {}
    for p in sorted(glob.glob(f"{d}/hf/data/*.parquet")):
        for row in pq.read_table(p, columns=["image"]).column("image").to_pylist():
            imgs[row["path"]] = row["bytes"]

    def recs(split, site, keep=None):
        out = []
        for a in json.load(open(f"{d}/ChestX_Det_{split}.json")):
            if keep is not None and a["file_name"] not in keep:
                continue
            rid = a["file_name"][:-4]
            img = Image.open(io.BytesIO(imgs[a["file_name"]]))
            path, (w, h), s = save(site, rid, img)
            ev = {}
            for sym, b in zip(a["syms"], a["boxes"]):
                b = [v * s for v in b]
                if CHESTXDET_TO_F.get(sym):
                    add(ev, CHESTXDET_TO_F[sym], b)
                if sym in OPACITY_SOURCES:
                    add(ev, "opacity", b)
            out.append(dict(id=rid, path=path, w=w, h=h, has_boxes=True, evidence=ev, verifiable=FINDINGS,
                            labels={}, meta=dict(split=split, syms=sorted(set(a["syms"])), source="NIH ChestX-ray14")))
        return out

    write("chestxdet", recs("test", "chestxdet"))
    train = [a["file_name"] for a in json.load(open(f"{d}/ChestX_Det_train.json"))]
    write("chestxdet_cal", recs("train", "chestxdet_cal", set(random.Random(0).sample(train, 500))))


# ---- Shenzhen -----------------------------------------------------------------------------------------------
# Lesion types in the Shenzhen annotations (Data 2022) -> shared findings; all of them also support "opacity"
# except purely pleural ones.
SZ_MAP = {"infiltrate": "consolidation", "consolidation": "consolidation", "pneumonia": "consolidation",
          "nodule": "nodule_mass", "mass": "nodule_mass", "cavity": "nodule_mass",
          "pleural_effusion": "effusion", "effusion": "effusion",
          "thickening": "pleural_thickening", "pleural": "pleural_thickening",
          "atelectasis": "atelectasis", "collapse": "atelectasis", "pneumothorax": "pneumothorax"}
SZ_NOT_OPACITY = {"effusion", "pleural_thickening", "pneumothorax"}


SZ_OPACITY_ONLY = {"Calcification_(other_than_nodule&lymphnod)", "Calcified_lymph_node", "Linear_Density", "Miliary"}
SZ_IGNORE = {"Adenopathy", "Retraction", "Other", "Unknown"}      # not a lung finding in the shared space


def sz_finding(label):
    if label in SZ_OPACITY_ONLY:
        return None
    l = label.lower()
    for k, f in SZ_MAP.items():
        if k in l:
            return f
    return None


def shenzhen():
    d = f"{DATA}/shenzhen"
    pngs = sorted(glob.glob(f"{d}/CXR_png/*.png"))
    if not pngs:
        print("shenzhen: CXR_png not downloaded yet"); return
    types = collections.Counter()
    out = []
    for p in pngs:
        rid = os.path.basename(p)[:-4]
        tb = int(rid.endswith("_1"))
        img = Image.open(p)
        path, (w, h), s = save("shenzhen", rid, img)
        ev = {}
        for reader in ("Annotations", "Annotations-2"):
            for j in glob.glob(f"{d}/{reader}/Annotations_json/SeparateFiles/{rid}.json"):
                for v in json.load(open(j)).values():
                    for reg in v.get("regions", []):
                        sa = reg["shape_attributes"]
                        xs, ys = sa.get("all_points_x"), sa.get("all_points_y")
                        if sa.get("name") == "rect":
                            xs, ys = [sa["x"], sa["x"] + sa["width"]], [sa["y"], sa["y"] + sa["height"]]
                        if not xs:
                            continue
                        box = [min(xs) * s, min(ys) * s, max(xs) * s, max(ys) * s]
                        for lab in reg.get("region_attributes", {}):
                            types[lab] += 1
                            if lab in SZ_IGNORE:
                                continue
                            f = sz_finding(lab)
                            if f:
                                add(ev, f, box)
                            if f not in SZ_NOT_OPACITY:
                                add(ev, "opacity", box)
        reading = ""
        for t in glob.glob(f"{d}/ClinicalReadings/{rid}.txt"):
            reading = " ".join(open(t, errors="ignore").read().split())
        # Normal films: every finding is absent. TB films: lesions were outlined exhaustively, but the heart and
        # bones were not reviewed, so cardiomegaly and fracture claims cannot be checked there.
        ver = FINDINGS if not tb else [f for f in FINDINGS if f not in ("cardiomegaly", "fracture")]
        out.append(dict(id=rid, path=path, w=w, h=h, has_boxes=True, evidence=ev, verifiable=ver,
                        labels=dict(tb=tb, abnormal=tb), meta=dict(reading=reading, source="Shenzhen No.3 Hospital")))
    print("shenzhen lesion types:", dict(types))
    write("shenzhen", out)


# ---- label-only sites ---------------------------------------------------------------------------------------
def montgomery():
    d = f"{DATA}/montgomery"
    out = []
    for p in sorted(glob.glob(f"{d}/CXR_png/*.png")):
        rid = os.path.basename(p)[:-4]; tb = int(rid.endswith("_1"))
        path, (w, h), _ = save("montgomery", rid, Image.open(p))
        reading = ""
        for t in glob.glob(f"{d}/ClinicalReadings/{rid}.txt"):
            reading = " ".join(open(t, errors="ignore").read().split())
        out.append(dict(id=rid, path=path, w=w, h=h, has_boxes=False, labels=dict(tb=tb, abnormal=tb),
                        meta=dict(reading=reading, source="Montgomery County")))
    if out:
        write("montgomery", out)


def india_mendeley():
    d = f"{DATA}/india_mendeley"
    out = []
    for cls, lab in (("covid", dict(covid=1, abnormal=1)), ("non-covid", dict(covid=0, abnormal=1)),
                     ("normal", dict(covid=0, abnormal=0))):
        for p in sorted(glob.glob(f"{d}/{cls}/*")):
            rid = cls + "_" + re.sub(r"\W+", "_", os.path.splitext(os.path.basename(p))[0]).strip("_")
            path, (w, h), _ = save("india_mendeley", rid, Image.open(p))
            out.append(dict(id=rid, path=path, w=w, h=h, has_boxes=False, labels=lab,
                            meta=dict(cls=cls, source="KGMU Lucknow / UPUMS Saifai / GMC Kota")))
    write("india_mendeley", out)


def dicom_image(p):
    """DICOM -> 8-bit image: window to the 0.5-99.5 percentile range; invert MONOCHROME1 so bone is bright."""
    import pydicom
    ds = pydicom.dcmread(p)
    x = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(x, [0.5, 99.5])
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        x = 1 - x
    return Image.fromarray((x * 255).round().astype(np.uint8))


def india_nitrd():
    d = f"{DATA}/india_nitrd"
    out = []
    files = sorted(glob.glob(f"{d}/Dataset */*/*.jpg") + glob.glob(f"{d}/Dataset */*/*.dcm"))
    for p in files:                        # Dataset A: 8-bit JPEG (px/nx); Dataset B: 14-bit DICOM (pd/nd)
        ds, part, name = p.split("/")[-3:]
        rid = f"{ds[-1]}_{part.split()[0].lower()}_{os.path.splitext(name)[0]}"
        tb = int(name.startswith("p"))
        path, (w, h), _ = save("india_nitrd", rid, dicom_image(p) if p.endswith(".dcm") else Image.open(p))
        out.append(dict(id=rid, path=path, w=w, h=h, has_boxes=False, labels=dict(tb=tb, abnormal=tb),
                        meta=dict(dataset=ds, source="NITRD New Delhi")))
    write("india_nitrd", out)


def frontal_prob():
    """BiomedCLIP zero-shot P(frontal) vs P(lateral), on CPU (IU-Xray has no view metadata)."""
    import torch, open_clip
    name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, pre = open_clip.create_model_from_pretrained(name)
    tok = open_clip.get_tokenizer(name)
    model.eval()
    with torch.no_grad():
        t = model.encode_text(tok(["a frontal (PA or AP) chest X-ray", "a lateral chest X-ray"]))
        t = t / t.norm(dim=-1, keepdim=True)

    def f(path):
        with torch.no_grad():
            v = model.encode_image(pre(Image.open(path).convert("RGB")).unsqueeze(0))
            v = v / v.norm(dim=-1, keepdim=True)
            return float((100 * v @ t.T).softmax(-1)[0, 0])
    return f


# ---- IU-Xray ------------------------------------------------------------------------------------------------
def iuxray(n=1000):
    import xml.etree.ElementTree as ET
    d = f"{DATA}/iuxray"
    xmls = sorted(glob.glob(f"{d}/NLMCXR_reports/**/*.xml", recursive=True))
    pngs = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{d}/NLMCXR_png/**/*.png", recursive=True)}
    if not xmls or not pngs:
        print("iuxray: not downloaded yet"); return
    cands = []
    for x in xmls:
        t = ET.parse(x).getroot()
        txt = {a.get("Label"): (a.text or "").strip() for a in t.iter("AbstractText")}
        findings = txt.get("FINDINGS", "")
        ims = [i.get("id") for i in t.iter("parentImage") if i.get("id") in pngs]
        if not findings or not ims:
            continue
        cands.append((ims, findings, txt.get("IMPRESSION", "")))
    random.Random(0).shuffle(cands)
    # frontal view: highest BiomedCLIP zero-shot P(frontal) among the study's images; skip studies without one
    pf, picked = frontal_prob(), []
    for ims, findings, imp in cands:
        p = {i: pf(pngs[i]) for i in ims}
        best = max(p, key=p.get)
        if p[best] >= 0.5:
            picked.append((best, findings, imp))
        if len(picked) == n:
            break
    out = []
    for rid, findings, imp in sorted(picked):
        path, (w, h), _ = save("iuxray", rid, Image.open(pngs[rid]))
        out.append(dict(id=rid, path=path, w=w, h=h, has_boxes=False, labels={}, report=findings,
                        meta=dict(impression=imp, source="Indiana University")))
    write("iuxray", out)


SITES = dict(chestxdet=chestxdet, shenzhen=shenzhen, montgomery=montgomery, iuxray=iuxray,
             india_mendeley=india_mendeley, india_nitrd=india_nitrd)

if __name__ == "__main__":
    for s in sys.argv[1:] or SITES:
        SITES[s]()
