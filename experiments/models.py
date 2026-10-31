"""Uniform wrappers around the VLMs. Every wrapper offers:

  report(img)              -> dict(text, tokens=[(str, logprob)])   greedy findings section
  samples(img, k, temp)    -> [text]                                 k sampled findings sections
  locate(img, finding, k)  -> [[box,...]]                            greedy + k sampled box lists, boxes in [0,1] xyxy
  p_yes(img, question)     -> float                                  P(yes) / (P(yes)+P(no)) for the first answer token

img is a PIL RGB image. CheXagent-2 needs transformers==4.40 (separate venv); the Qwen-family models use 4.57.
"""
import os, re, json, math, tempfile
import torch

HF = os.environ.get("HF_HOME", "/mnt/hdd/cxr_vlm/hf")
GEN = dict(max_new_tokens=320, repetition_penalty=1.05)


def _boxes_from_numbers(text, scale):
    out = []
    for m in re.finditer(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", text):
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        b = [min(x1, x2) / scale[0], min(y1, y2) / scale[1], max(x1, x2) / scale[0], max(y1, y2) / scale[1]]
        b = [min(max(v, 0.0), 1.0) for v in b]
        if b[2] > b[0] and b[3] > b[1]:
            out.append(b)
    return out


def _token_logprobs(model, tok, out, n_in):
    scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
    ids = out.sequences[0, n_in:]
    return [(tok.decode([int(i)]), float(s)) for i, s in zip(ids, scores) if int(i) not in tok.all_special_ids]


def _yes_no_ids(tok):
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
    return ids(["Yes", "yes", " Yes", " yes"]), ids(["No", "no", " No", " no"])


class QwenVL:
    """Qwen2.5-VL-7B-Instruct and Lingshu-7B (same architecture). Boxes are absolute pixels of the resized input."""
    REPORT = ("You are an expert radiologist. Write the findings section of a radiology report for this chest X-ray. "
              "Describe only what is visible in the image.")
    LOCATE = "Locate every region showing {f} in this chest X-ray. Output the bounding box coordinates in JSON format."

    def __init__(self, repo, bits=4):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True) if bits == 4 else None
        if os.environ.get("HF_HUB_OFFLINE") == "1":   # local path: transformers 4.57 otherwise calls the Hub API
            from huggingface_hub import snapshot_download        # (Mistral-regex check) even in offline mode
            repo = snapshot_download(repo, local_files_only=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            repo, torch_dtype=torch.bfloat16, quantization_config=q, device_map="cuda:0", attn_implementation="sdpa").eval()
        self.proc = AutoProcessor.from_pretrained(repo, min_pixels=256 * 28 * 28, max_pixels=1024 * 1024)
        self.tok = self.proc.tokenizer
        self.yes, self.no = _yes_no_ids(self.tok)

    def _inputs(self, img, prompt):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        x = self.proc(text=[text], images=[img], return_tensors="pt").to(self.model.device)
        t, gh, gw = x["image_grid_thw"][0].tolist()
        return x, (gw * 14, gh * 14)

    @torch.no_grad()
    def _gen(self, img, prompt, k=1, temp=None, scores=False, max_new_tokens=None):
        x, size = self._inputs(img, prompt)
        kw = dict(GEN, do_sample=temp is not None, num_return_sequences=k)
        if max_new_tokens: kw["max_new_tokens"] = max_new_tokens
        if temp is not None: kw.update(temperature=temp, top_p=0.95, top_k=0)   # top_k=0: override Lingshu's top_k=1
        else: kw.update(temperature=None, top_p=None, top_k=None)
        out = self.model.generate(**x, **kw, output_scores=scores, return_dict_in_generate=True)
        n = x["input_ids"].shape[1]
        texts = [self.tok.decode(s[n:], skip_special_tokens=True).strip() for s in out.sequences]
        return texts, out, n, size

    def report(self, img):
        texts, out, n, _ = self._gen(img, self.REPORT, scores=True)
        return dict(text=texts[0], tokens=_token_logprobs(self.model, self.tok, out, n))

    def samples(self, img, k, temp=0.7):
        return self._gen(img, self.REPORT, k=k, temp=temp)[0]

    def locate(self, img, finding, k=3, temp=0.7):
        g, _, _, size = self._gen(img, self.LOCATE.format(f=finding), max_new_tokens=160)
        s = self._gen(img, self.LOCATE.format(f=finding), k=k, temp=temp, max_new_tokens=160)[0] if k else []
        return [_boxes_from_numbers(t, size) for t in g + s], g + s

    @torch.no_grad()
    def p_yes(self, img, question):
        return self.p_yes_many(img, [question])[0]

    @torch.no_grad()
    def p_yes_many(self, img, questions, batch=6):
        """All yes/no questions about one image in left-padded batches (one pass instead of one per question)."""
        out = []
        for i in range(0, len(questions), batch):
            qs = questions[i:i + batch]
            texts = [self.proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text", "text": q + " Answer with yes or no."}]}],
                tokenize=False, add_generation_prompt=True) for q in qs]
            self.tok.padding_side = "left"
            x = self.proc(text=texts, images=[img] * len(qs), padding=True, return_tensors="pt").to(self.model.device)
            p = torch.softmax(self.model(**x).logits[:, -1].float(), -1)
            for row in p:
                y, n = row[self.yes].sum().item(), row[self.no].sum().item()
                out.append(y / (y + n) if y + n > 0 else 0.5)
        return out


class CheXagent2:
    """StanfordAIMI/CheXagent-2-3b (bf16). Boxes come as <|box|>(x1,y1),(x2,y2)<|/box|> on a 0-100 grid."""
    REPORT = 'Given the indication: "Routine chest radiograph.", write a structured findings section for the CXR.'
    LOCATE = "Locate areas in the chest X-ray where {f} are present, using bounding box coordinates"

    def __init__(self, repo="StanfordAIMI/CheXagent-2-3b"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        # as on the model card: load, then cast (loading with torch_dtype leaves some fp32 buffers behind)
        # (cast on the CPU before moving, so the GPU never holds the fp32 copy: ~7 GB instead of ~12 GB)
        self.model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        self.model = self.model.to(torch.bfloat16).to("cuda:0").eval()
        torch.cuda.empty_cache()
        self.yes, self.no = _yes_no_ids(self.tok)
        self.tmp = tempfile.mkdtemp(prefix="chexagent_")

    def _ids(self, img, prompt):
        path = os.path.join(self.tmp, "x.png"); img.save(path)
        q = self.tok.from_list_format([{"image": path}, {"text": prompt}])
        conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": q}]
        return self.tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to("cuda:0")

    @torch.no_grad()
    def _gen(self, img, prompt, k=1, temp=None, scores=False, max_new_tokens=None):
        ids = self._ids(img, prompt)
        kw = dict(GEN, do_sample=temp is not None, num_return_sequences=k, use_cache=True, num_beams=1)
        if max_new_tokens: kw["max_new_tokens"] = max_new_tokens
        if temp is not None: kw.update(temperature=temp, top_p=0.95)
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=self.tok.eos_token_id,
                                  **kw, output_scores=scores, return_dict_in_generate=True)
        n = ids.shape[1]
        texts = [self.tok.decode(s[n:]).replace("<|endoftext|>", "").replace("<|im_end|>", "").strip() for s in out.sequences]
        return texts, out, n

    @staticmethod
    def _clean(t):
        """Grounded output '<|ref|> X <|/ref|> <|box|>(..)<|/box|>' -> 'X.' so CheXbert reads plain sentences."""
        t = re.sub(r"<\|box\|>.*?<\|/box\|>", "", t)
        t = re.sub(r"<\|ref\|>\s*(.*?)\s*<\|/ref\|>", r"\1. ", t)
        return re.sub(r"\s+", " ", t).strip()

    def report(self, img):
        texts, out, n = self._gen(img, self.REPORT, scores=True)
        return dict(text=self._clean(texts[0]), raw=texts[0], report_boxes=self._boxes(texts[0]),
                    tokens=_token_logprobs(self.model, self.tok, out, n))

    def samples(self, img, k, temp=0.7):
        return [self._clean(t) for t in self._gen(img, self.REPORT, k=k, temp=temp)[0]]

    @staticmethod
    def _boxes(text):
        out = []
        for m in re.finditer(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text):
            x1, y1, x2, y2 = (int(v) / 100 for v in m.groups())
            if x2 > x1 and y2 > y1:
                out.append([x1, y1, x2, y2])
        return out

    def locate(self, img, finding, k=3, temp=0.7):
        g = self._gen(img, self.LOCATE.format(f=finding), max_new_tokens=160)[0]
        s = self._gen(img, self.LOCATE.format(f=finding), k=k, temp=temp, max_new_tokens=160)[0] if k else []
        return [self._boxes(t) for t in g + s], g + s

    @torch.no_grad()
    def p_yes(self, img, question):
        logits = self.model(self._ids(img, question + " Answer with yes or no.")).logits
        p = torch.softmax(logits[0, -1].float(), -1)
        y, n = p[self.yes].sum().item(), p[self.no].sum().item()
        return y / (y + n) if y + n > 0 else 0.5

    def p_yes_many(self, img, questions):
        return [self.p_yes(img, q) for q in questions]      # its remote code has no batched image input


class QwenVLLM:
    """Qwen2.5-VL / Lingshu served by vLLM with FP8 weights (venv_vllm). Same prompts and outputs as QwenVL.
    first_pass_many() sends the report, the K samples and all yes/no questions of several images as one batch."""
    REPORT, LOCATE = QwenVL.REPORT, QwenVL.LOCATE

    def __init__(self, repo, quant="fp8", mem=0.80):
        # batch-invariant kernels: the same image gives the same greedy output whatever else is in the batch
        # (tested: 20/20 identical reports and P(yes) at 1 vs 8 images per call, no speed cost)
        os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
        from vllm import LLM
        from transformers import AutoProcessor
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            from huggingface_hub import snapshot_download
            repo = snapshot_download(repo, local_files_only=True)
        px = dict(min_pixels=256 * 28 * 28, max_pixels=1024 * 1024)
        # generation_config="vllm": ignore the checkpoint's sampling defaults (Lingshu ships top_k=1, which made its
        # "samples" near-greedy under transformers); only the settings below apply
        self.llm = LLM(repo, quantization=quant, max_model_len=4096, gpu_memory_utilization=mem, seed=0,
                       limit_mm_per_prompt={"image": 1}, mm_processor_kwargs=px, generation_config="vllm")
        self.proc = AutoProcessor.from_pretrained(repo, **px)
        self.tok = self.llm.get_tokenizer()
        self.yes, self.no = _yes_no_ids(self.tok)
        self.special = set(self.tok.all_special_ids)

    def _p(self, img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return {"prompt": self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                "multi_modal_data": {"image": img}}

    def _size(self, img):        # boxes come in pixels of the resized model input
        t, gh, gw = self.proc.image_processor(images=[img], return_tensors="pt")["image_grid_thw"][0].tolist()
        return gw * 14, gh * 14

    def _sp(self, **kw):
        from vllm import SamplingParams
        return SamplingParams(repetition_penalty=GEN["repetition_penalty"], **kw)

    def _pyes(self, out):
        lp = out.outputs[0].logprobs[0]                        # top-20 first-token log-probs
        y = sum(math.exp(lp[i].logprob) for i in self.yes if i in lp)
        n = sum(math.exp(lp[i].logprob) for i in self.no if i in lp)
        return y / (y + n) if y + n > 0 else 0.5

    def first_pass_many(self, imgs, k, temp, questions, seeds=None):
        """Report, K samples and every yes/no question for several images in one vLLM call.
        seeds: one per image, so each image's samples are reproducible whatever the batch."""
        prompts, params = [], []
        g_sp = self._sp(temperature=0, max_tokens=GEN["max_new_tokens"], logprobs=0)
        q_sp = self._sp(temperature=0, max_tokens=1, logprobs=20)
        for i, img in enumerate(imgs):
            prompts.append(self._p(img, self.REPORT)); params.append(g_sp)
            if k:
                prompts.append(self._p(img, self.REPORT))
                params.append(self._sp(n=k, temperature=temp, top_p=0.95, max_tokens=GEN["max_new_tokens"],
                                       seed=seeds[i] if seeds else None))
            prompts += [self._p(img, q + " Answer with yes or no.") for q in questions]; params += [q_sp] * len(questions)
        outs, res, per = self.llm.generate(prompts, params, use_tqdm=False), [], 1 + bool(k) + len(questions)
        for i in range(len(imgs)):
            o = outs[i * per:(i + 1) * per]
            g = o[0].outputs[0]
            rep = dict(text=g.text.strip(), tokens=[(self.tok.decode([t]), lp[t].logprob)
                                                    for t, lp in zip(g.token_ids, g.logprobs) if t not in self.special])
            samples = [x.text.strip() for x in o[1].outputs] if k else []
            res.append((rep, samples, [self._pyes(x) for x in o[1 + bool(k):]]))
        return res

    def locate_batch(self, items, k=3, temp=0.7, seeds=None):
        """items: [(img, [finding names])] -> one dict per item {name: (box lists, raw texts)}, greedy + k samples."""
        prompts, params, keys = [], [], []
        g_sp = self._sp(temperature=0, max_tokens=160)
        for j, (img, fs) in enumerate(items):
            for n, f in enumerate(fs):
                p = self._p(img, self.LOCATE.format(f=f))
                prompts.append(p); params.append(g_sp); keys.append((j, f, "g"))
                if k:
                    prompts.append(p); keys.append((j, f, "s"))
                    params.append(self._sp(n=k, temperature=temp, top_p=0.95, max_tokens=160,
                                           seed=seeds[j] * 31 + n if seeds else None))
        outs = self.llm.generate(prompts, params, use_tqdm=False) if prompts else []
        raw = [{f: [] for f in fs} for _, fs in items]
        for (j, f, kind), o in zip(keys, outs):
            raw[j][f] += [x.text.strip() for x in o.outputs]        # greedy first, then its k samples
        res = []
        for (img, fs), rj in zip(items, raw):
            size = self._size(img)
            res.append({f: ([_boxes_from_numbers(t, size) for t in rj[f]], rj[f]) for f in fs})
        return res

    def locate(self, img, finding, k=3, temp=0.7):
        return self.locate_batch([(img, [finding])], k, temp)[0][finding]


MODELS = {
    # Qwen family: vLLM + FP8 in venv_vllm (5.1x faster than 4-bit transformers on 20 matched images)
    "qwen25vl7b": lambda: QwenVLLM("Qwen/Qwen2.5-VL-7B-Instruct"),
    "lingshu7b": lambda: QwenVLLM("lingshu-medical-mllm/Lingshu-7B"),
    "chexagent2": lambda: CheXagent2(),
    # the earlier 4-bit transformers backend (runs archived in runs_nf4/)
    "qwen25vl7b_nf4": lambda: QwenVL("Qwen/Qwen2.5-VL-7B-Instruct"),
    "lingshu7b_nf4": lambda: QwenVL("lingshu-medical-mllm/Lingshu-7B"),
}
