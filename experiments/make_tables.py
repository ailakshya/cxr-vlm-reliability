"""Turn results/*.json (from evaluate.py) into LaTeX tables and PDF figures for the paper.

  python make_tables.py [results_dir] [out_dir]      default: $CXR_ROOT/results  ->  $CXR_ROOT/paper_out

Writes tab_esp.tex, tab_calib.tex, tab_ablation.tex, tab_probes.tex, fig_corruption.pdf, fig_local.pdf.
Missing values print as "--", so the script can run at any point of the experiment.
"""
import glob, json, os, sys
import numpy as np

ROOT = os.environ.get("CXR_ROOT", "/mnt/hdd/cxr_vlm")
RES = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/paper_out"
MODELS = [("chexagent2", "CheXagent-2"), ("qwen25vl7b", "Qwen2.5-VL"), ("lingshu7b", "Lingshu"),
          ("medgemma15", "MedGemma 1.5"), ("maira2", "MAIRA-2")]
SIGNALS = [("conf_logprob", "Sentence prob."), ("conf_consistency", "Sample agreement"), ("conf_pyes", "Yes/no probe"),
           ("conf_evidence", "Box agreement"), ("conf_sgf", "Self-grounding")]
FAMILIES = ["noise", "blur", "jpeg", "contrast", "rotate"]

R = {}
for p in glob.glob(f"{RES}/*.json"):
    r = json.load(open(p))
    R[(r["model"], r["site"], r["cond"])] = r


def g(m, site, cond, key):
    v = R.get((m, site, cond), {}).get(key)
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def pct(v, d=1):
    return "--" if v is None else f"{100 * v:.{d}f}"


def num(v, d=2):
    return "--" if v is None else f"{v:.{d}f}"


def present(models):
    return [(m, n) for m, n in models if any(k[0] == m for k in R)]


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/{name}", "w").write(text)
    print("wrote", f"{OUT}/{name}")


def tab_esp():
    rows = []
    for m, n in present(MODELS):
        cells = []
        for site in ("chestxdet", "shenzhen"):
            ci = g(m, site, "clean", "ESP_text_CI")
            ci = f" \\tiny[{pct(ci[0], 0)}--{pct(ci[1], 0)}]" if ci else ""
            cells += [pct(g(m, site, "clean", "ESP_text")) + ci, pct(g(m, site, "clean", "ESP_iou0.3")),
                      pct(g(m, site, "clean", "ER_text")), num(g(m, site, "clean", "mIoU"))]
        rows.append(f"{n} & " + " & ".join(cells) + r" \\")
    write("tab_esp.tex", "\n".join(rows) + "\n")


def tab_calib():
    rows = []
    for m, n in present(MODELS):
        for site, sn in (("chestxdet", "ChestX-Det"), ("shenzhen", "Shenzhen")):
            cells = [f"{num(g(m, site, 'clean', s + '__AUROC'))} / {num(g(m, site, 'clean', s + '__AURC'))}"
                     for s, _ in SIGNALS]
            rows.append(f"{n} & {sn} & " + " & ".join(cells) + r" \\")
    write("tab_calib.tex", "\n".join(rows) + "\n")


def tab_ablation():
    rows = []
    for m, n in present(MODELS):
        for site, sn in (("chestxdet", "ChestX-Det"), ("shenzhen", "Shenzhen")):
            ci = g(m, site, "clean", "abl_EDI_pyes_CI")
            rows.append(f"{n} & {sn} & {g(m, site, 'clean', 'abl_n_claimed') or '--'} & "
                        f"{pct(g(m, site, 'clean', 'abl_claim_drop_ev'))} & {pct(g(m, site, 'clean', 'abl_claim_drop_rnd'))} & "
                        f"{pct(g(m, site, 'clean', 'abl_EDI_claims'))} & {num(g(m, site, 'clean', 'abl_EDI_pyes'), 3)}"
                        + (f" [{num(ci[0], 3)}, {num(ci[1], 3)}]" if ci else "") + r" \\")
    write("tab_ablation.tex", "\n".join(rows) + "\n")


def tab_probes():
    rows = []
    for m, n in present(MODELS):
        for site, sn, key in (("montgomery", "Montgomery (USA)", "tb"), ("shenzhen", "Shenzhen (China)", "tb"),
                              ("india_mendeley", "Mendeley (India)", "abnormal"), ("india_nitrd", "NITRD (India)", "tb")):
            rows.append(f"{n} & {sn} & {key.upper() if key == 'tb' else 'Abnormal'} & "
                        f"{num(g(m, site, 'clean', key + '_AUROC'))} & {pct(g(m, site, 'clean', key + '_acc'))} & "
                        f"{num(g(m, site, 'clean', key + '_ECE'))} & {pct(g(m, site, 'clean', key + '_claim_acc'))}" + r" \\")
    write("tab_probes.tex", "\n".join(rows) + "\n")


def fig_corruption():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(FAMILIES), figsize=(7.2, 3.6), sharey="row")
    for j, fam in enumerate(FAMILIES):
        for m, n in present(MODELS):
            for i, (key, lab) in enumerate((("UCR_text", "Unsupported-claim rate"), ("quality_mention", "Quality mentioned"))):
                xs, ys = [], []
                for sev, cond in ((0, "clean"), (1, f"{fam}_1"), (3, f"{fam}_3")):
                    v = g(m, "chestxdet", cond, key)
                    if v is not None:
                        xs.append(sev); ys.append(v)
                if xs:
                    ax[i, j].plot(xs, ys, marker="o", ms=3, label=n)
                ax[i, 0].set_ylabel(lab, fontsize=7)
        ax[0, j].set_title(fam, fontsize=8); ax[1, j].set_xticks([0, 1, 3], ["clean", "mild", "severe"], fontsize=6)
        ax[0, j].tick_params(labelsize=6); ax[1, j].tick_params(labelsize=6)
    ax[0, 0].legend(fontsize=6)
    fig.tight_layout(); os.makedirs(OUT, exist_ok=True); fig.savefig(f"{OUT}/fig_corruption.pdf"); print("wrote fig_corruption.pdf")


def fig_local():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for m, n in present(MODELS):
        for s, sl, ls in (("conf_consistency", "agreement", "--"), ("conf_sgf", "self-grounding", "-")):
            ms, cov, kept = [], [], []
            for k in (0, 10, 25, 50):
                key = f"conf_{s}_a0.1_coverage" if k == 0 else f"local{k}_{s}_a0.1_coverage"
                c = g(m, "shenzhen", "clean", key); kp = g(m, "shenzhen", "clean", key.replace("coverage", "kept"))
                if c is not None:
                    ms.append(k); cov.append(c); kept.append(kp)
            if ms:
                ax[0].plot(ms, cov, ls, marker="o", ms=3, label=f"{n}, {sl}")
                ax[1].plot(ms, kept, ls, marker="o", ms=3)
    ax[0].axhline(0.9, color="grey", lw=0.8); ax[0].set_ylabel("coverage (target 0.9)", fontsize=7)
    ax[1].set_ylabel("claims kept", fontsize=7)
    for a in ax:
        a.set_xlabel("local labelled images m (0 = calibrated on ChestX-Det)", fontsize=7); a.tick_params(labelsize=6)
    ax[0].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_local.pdf"); print("wrote fig_local.pdf")


if __name__ == "__main__":
    tab_esp(); tab_calib(); tab_ablation(); tab_probes(); fig_corruption(); fig_local()
