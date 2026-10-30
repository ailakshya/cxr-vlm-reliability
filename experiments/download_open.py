"""Download the open datasets into $DATA (default /mnt/hdd/cxr_vlm/data). Re-running skips finished files.

  iuxray/      Open-i Indiana PNGs + XML reports
  chestxdet/   ChestX-Det images (HF mirror natealberti/ChestX-Det) + official JSON boxes (Deepwise GitHub)
  shenzhen/    NLM Shenzhen CXR_png, ClinicalReadings, Annotations, consensus ROI boxes
  montgomery/  NLM Montgomery CXR_png, ClinicalReadings
  india_mendeley/  KGMU / UPUMS / GMC Kota COVID, non-COVID, normal (DOI 10.17632/4n66brtp4j.1)
  india_nitrd/ NITRD New Delhi TB Dataset A/B (SourceForge tbxpredict)
"""
import os, re, sys, tarfile, zipfile, subprocess, urllib.parse, urllib.request

DATA = os.environ.get("DATA", "/mnt/hdd/cxr_vlm/data")
UA = {"User-Agent": "Mozilla/5.0 (research download; lakshya)"}


def fetch(url, dst, ua=UA["User-Agent"]):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("GET", url, flush=True)
    subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "5", "-A", ua, "-o", dst + ".part", url], check=True)
    os.replace(dst + ".part", dst)
    return dst


def page(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "ignore")


def nlm_crawl(url, dst):
    """NLM 'Index of' pages: links are single-quoted; folders end with /index.html."""
    for href in re.findall(r"<a href='([^']+)'", page(url)):
        full = urllib.parse.urljoin(url, href)
        name = urllib.parse.unquote(href.split("/")[0])
        if href.endswith("index.html"):
            nlm_crawl(full, os.path.join(dst, name))
        elif not href.startswith(("?", "/", "..")):
            fetch(full, os.path.join(dst, name))


def unpack(path, dst):
    flag = os.path.join(dst, ".unpacked_" + os.path.basename(path))
    if os.path.exists(flag):
        return
    os.makedirs(dst, exist_ok=True)
    (tarfile.open(path) if path.endswith((".tgz", ".tar.gz")) else zipfile.ZipFile(path)).extractall(dst)
    open(flag, "w").close()


def iuxray():
    d = f"{DATA}/iuxray"
    for f in ("NLMCXR_png.tgz", "NLMCXR_reports.tgz"):
        unpack(fetch(f"https://openi.nlm.nih.gov/imgs/collections/{f}", f"{d}/raw/{f}"), f"{d}/{f.split('.')[0]}")


def chestxdet():
    d = f"{DATA}/chestxdet"
    for f in ("ChestX_Det_train.json", "ChestX_Det_test.json"):
        fetch(f"https://raw.githubusercontent.com/Deepwise-AILab/ChestX-Det-Dataset/main/{f}", f"{d}/{f}")
    for f in ("README.md", "id2label.json", "data/test-00000-of-00001.parquet",
              "data/train-00000-of-00003.parquet", "data/train-00001-of-00003.parquet", "data/train-00002-of-00003.parquet"):
        fetch(f"https://huggingface.co/datasets/natealberti/ChestX-Det/resolve/main/{f}", f"{d}/hf/{f}")


def nlm():
    base = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"
    nlm_crawl(f"{base}/Shenzhen-Hospital-CXR-Set/index.html", f"{DATA}/shenzhen")
    nlm_crawl(f"{base}/Montgomery-County-CXR-Set/MontgomerySet/index.html", f"{DATA}/montgomery")


def india_mendeley():
    import json
    d = f"{DATA}/india_mendeley"
    files = json.loads(page("https://data.mendeley.com/public-api/datasets/4n66brtp4j/files?folder_id=root&version=1"))
    for f in files:
        unpack(fetch(f["content_details"]["download_url"], f"{d}/raw/{f['filename']}"), d)


def india_nitrd():
    # SourceForge blocks the HTML file browser (403); its RSS feed lists every file.
    d, pre = f"{DATA}/india_nitrd", "https://sourceforge.net/projects/tbxpredict/files/data/"
    # Each feed shows at most 100 items, so read one feed per folder and fill gaps from the nxK/pxK naming.
    names = {"label_information.txt"}
    for ds in ("Dataset A", "Dataset B"):
        for part in ("Training Data", "Testing Data"):
            folder = f"{ds}/{part}"
            rss = page("https://sourceforge.net/projects/tbxpredict/rss?path=/data/" + urllib.parse.quote(folder))
            got = [urllib.parse.unquote(u[len(pre):-len("/download")]) for u in re.findall(r"<link>([^<]+/download)</link>", rss)]
            got = [g for g in got if g.startswith(folder + "/")]
            names.update(got)
            for cls in ("nx", "px"):
                ks = [int(m) for g in got for m in re.findall(rf"/{cls}(\d+)\.jpg$", g)]
                if ks:
                    names.update(f"{folder}/{cls}{k}.jpg" for k in range(min(ks), max(ks) + 1))
    for n in sorted(names):   # the /download links serve an HTML page; the mirror host serves the file itself
        fetch("https://master.dl.sourceforge.net/project/tbxpredict/data/" + urllib.parse.quote(n) + "?viasf=1", os.path.join(d, n), ua="Wget/1.21")  # browser UAs get an HTML page
    print(len(names), "files")


if __name__ == "__main__":
    jobs = dict(iuxray=iuxray, chestxdet=chestxdet, nlm=nlm, india_mendeley=india_mendeley, india_nitrd=india_nitrd)
    for k in (sys.argv[1:] or jobs):
        try:
            jobs[k]()
            print("DONE", k, flush=True)
        except Exception as e:
            print("FAILED", k, repr(e), flush=True)
