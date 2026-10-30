# VLMs for Chest X-ray Interpretation and Evidence Grounding (2023–2026): Models, Grounding Datasets, Grounding Evaluation

Verification legend: **[V]** = fetched/confirmed in this session (arXiv page, HF card, PhysioNet page, or search result pointing to the primary page). **[K]** = arXiv ID/fact from prior knowledge (training data, cutoff mid-2026), not re-fetched this session. The writer should spot-check [K] items before citing; none are believed wrong, but they were not re-verified. **UNVERIFIED** = could not confirm at all.
GPU memory figures marked "≈" are inferences (rule of thumb: bf16 ≈ 2 bytes/param + ~10–25% activation/KV overhead; 4-bit ≈ 0.6 bytes/param), not vendor numbers.

## Q1. CXR/medical VLMs: what they are, weights, size, grounding ability

### Takeaway
For a reliability-under-shift study with *evidence grounding*, the models that natively emit boxes for CXR are MAIRA-2 (7B, gated MSRLA, research only), MedGemma 1.5 4B (bbox anatomical localization added Apr 2026), RadVLM (PhysioNet credentialed), AnatomiX (2026, code+weights released), MedVersa (gated; segmentation/detection), and general models Qwen2.5-VL (native grounding JSON) and Gemini; most others (CheXagent-2, LLaVA-Med, LLaVA-Rad, RadFM, Lingshu, HuatuoGPT-Vision, XrayGPT, RaDialog, Med-Flamingo) are text-only and need post-hoc methods (attention/Grad-CAM, or prompting for coordinates) to get evidence regions. Contrastive models (BiomedCLIP, CheXzero, MedCLIP) give only similarity heatmaps.

### Cited Findings

**MAIRA-2 (Microsoft Health Futures)** [V]
- "MAIRA-2: Grounded Radiology Report Generation", Bannur, Bouzid, Castro, … Lungren, Wetscherek, Alvarez-Valle, Hyland (21 authors); arXiv:2406.04449 (v2 20 Sep 2024, added PadChest-GR results); arXiv page lists no venue — [arXiv](https://arxiv.org/abs/2406.04449)
- Introduces the task of *grounded report generation*; evaluated with RadFact, an LLM-based sentence-level framework for correctness/completeness that supports grounded reporting — [arXiv](https://arxiv.org/abs/2406.04449)
- HF `microsoft/maira-2`: MSRLA license, **gated** (accept research-only disclaimer), 7B params (F32 tensors in repo); architecture = frozen RAD-DINO-MAIRA-2 vision encoder + projection trained from scratch + fully fine-tuned vicuna-7b-v1.5 — [HF card](https://huggingface.co/microsoft/maira-2)
- Output format: list of (text, boxes) pairs; boxes as (x_tl, y_tl, x_br, y_br) relative to the cropped processed image, mapped back via `adjust_box_for_original_image_size`; ungrounded findings have no boxes — [HF card](https://huggingface.co/microsoft/maira-2)
- Training data: MIMIC-CXR, PadChest, private USMix, plus MS-CXR for grounded examples — [HF card](https://huggingface.co/microsoft/maira-2)
- Secondary sources report MAIRA-2 box mIoU ≈ 57.8 on MS-CXR phrase grounding, but MAIRA-2 trained on ~50% of MS-CXR, so comparisons are contaminated — [MedGround-R1 / search summary](https://arxiv.org/pdf/2507.02994); [emergentmind topic page](https://www.emergentmind.com/topics/maira-2) (secondary; not confirmed in primary paper this session)
- GPU: ≈16 GB bf16 (inference); F32 checkpoint ≈28 GB on disk.
- Predecessor MAIRA-1 arXiv:2311.13668 [K]; MAIRA-Seg (segmentation-mask-enhanced RRG) arXiv:2411.11362 [K].

**MedGemma / MedGemma 1.5 (Google)**
- MedGemma Technical Report, arXiv:2507.05201 (Jul 2025) [K]; models `google/medgemma-4b-it` (multimodal, Gemma 3 + MedSigLIP encoder), `google/medgemma-27b-it` / 27b-text; Health AI Developer Foundations (HAI-DEF) terms, **gated** on HF (accept terms) [K]. ≈10 GB bf16 for 4B; ≈55 GB bf16 for 27B (≈16–20 GB at 4-bit).
- "MedGemma 1.5 Technical Report", Sellergren, Gao, … (42 authors), arXiv:2604.05081, v1 6 Apr 2026, v2 1 May 2026 — [arXiv](https://arxiv.org/abs/2604.05081)
- MedGemma 1.5 4B adds: 3D CT/MRI, WSI pathology, **bounding-box anatomical localization**, multi-timepoint (longitudinal) CXR, medical documents; abstract reports "35% increase in IoU on chest X-rays" (relative vs absolute not stated in abstract, dataset not named there) and +4% macro accuracy on longitudinal CXR — [arXiv](https://arxiv.org/abs/2604.05081)
- A third-party blog gives MedGemma 1 ≈3% vs 1.5 ≈38% IoU (i.e., 35-point absolute) — [tinyweights blog](https://tinyweights.dev/posts/medgemma-1-5/) (secondary; check report tables)
- Box output is JSON; FiftyOne plugin pads to square so coords map back to original image — [Voxel51 docs](https://docs.voxel51.com/plugins/plugins_ecosystem/medgemma_1_5.html)
- Derivative: DobicVLM (arXiv:2607.18988) fine-tunes MedGemma-4B with GRPO + rule-based clinical rewards; 1,000 private image-report pairs, 69-case blinded expert eval — [arXiv](https://arxiv.org/abs/2607.18988v1)

**CheXagent (Stanford AIMI)**
- "CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation", Z. Chen, M. Varma, … C. Langlotz, et al.; arXiv:2401.12208; also introduced CheXinstruct (instruction data) and CheXbench (8 task types) [K for authors/benchmarks] — [HF card cites 2401.12208](https://huggingface.co/StanfordAIMI/CheXagent-2-3b)
- `StanfordAIMI/CheXagent-2-3b`: MIT license, not gated, 3B params; card tags text-generation/conversational only, no box output mentioned; HF collections "CheXagent" (9 items) and "CheXagent and its byproducts" (11) — [HF card](https://huggingface.co/StanfordAIMI/CheXagent-2-3b)
- Original `StanfordAIMI/CheXagent-8b` (v1) [K]; v1 paper describes some localization tasks in CheXbench (phrase grounding on MS-CXR) [K — verify]. GPU ≈8 GB bf16 (3B), ≈18 GB (8B).

**RadVLM (ETH Zurich / Krauthammer Lab)** [V]
- "RadVLM: A Multitask Conversational Vision-Language Model for Radiology", arXiv:2502.03333 (5 Feb 2025) — [arXiv](https://arxiv.org/pdf/2502.03333)
- >1M image-instruction pairs; tasks: report generation, abnormality classification, **visual grounding (bbox coordinates)**, multi-turn dialogue; claims SOTA on conversation and grounding — [arXiv](https://arxiv.org/html/2502.03333v1)
- Weights on PhysioNet (`radvlm-model`, **credentialed access**); authors also cite KrauthammerLab HF page — [PhysioNet](https://physionet.org/content/radvlm-model). Base/size not confirmed this session (believed LLaVA-OneVision-7B base [K]).

**AnatomiX (2026)** [V]
- "AnatomiX, an Anatomy-Aware Grounded Multimodal Large Language Model for Chest X-Ray Interpretation", Hashmi, Saeed, Lippert; arXiv:2601.03191 (v1 6 Jan 2026, v3 23 May 2026); venue not stated — [arXiv](https://arxiv.org/abs/2601.03191)
- Two-stage: detect anatomical structures + extract features, then LLM for phrase grounding, report generation, VQA; claims >25% improvement on anatomy grounding, phrase grounding, grounded diagnosis, grounded captioning; code + pretrained model released (project page aneesurhashmi.github.io/anatomix) — [arXiv](https://arxiv.org/abs/2601.03191)

**Other 2026 grounded-CXR preprints** [V, abstracts only]
- CheXGround: anatomical region tokens for grounded *longitudinal* CXR interpretation, arXiv:2608.30758 (Aug 2026) — [arXiv](https://arxiv.org/abs/2608.30758)
- GRCD: Grounded Region Change Detection for multi-finding CXR pairs, arXiv:2607.02719 — [arXiv](https://arxiv.org/pdf/2607.02719)
- Phrase-grounded fact-checking for generated CXR reports, arXiv:2509.21356 — [arXiv](https://arxiv.org/pdf/2509.21356)
- Look & Mark (eye fixations + boxes in MLLM RRG), arXiv:2505.22222 — [arXiv](https://arxiv.org/pdf/2505.22222)
- MMRad-22K structured multimodal evidence dataset for CXR RRG, arXiv:2602.12843 — [arXiv](https://arxiv.org/pdf/2602.12843)
- Context- and gaze-guided VLP for CXR, arXiv:2603.26049 — [arXiv](https://arxiv.org/pdf/2603.26049)
- MedGround-R1 (Qwen2.5-VL + spatial-semantic GRPO for medical grounding; reports Acc@IoU>0.5 and mIoU), arXiv:2507.02994 — [arXiv](https://arxiv.org/pdf/2507.02994)
- Knowledge-guided spatial prompts for medical visual grounding, arXiv:2604.01915 — [arXiv](https://arxiv.org/pdf/2604.01915)

**MedVersa (Harvard, Rajpurkar lab)**
- "A Generalist Learner for Multifaceted Medical Image Interpretation", H.-Y. Zhou, … P. Rajpurkar, arXiv:2405.07988 — [HF card](https://huggingface.co/hyzhou/MedVersa)
- `hyzhou/MedVersa`: **gated**, 7B, license not stated on card; returns 2D/3D segmentation masks, report generation; env files for A100/H100 — [HF card](https://huggingface.co/hyzhou/MedVersa). Paper describes CXR detection/grounding tasks [K].

**Lingshu (Alibaba DAMO)**
- arXiv:2506.07044; `lingshu-medical-mllm/Lingshu-7B` (listed 8B BF16) and Lingshu-32B; MIT; Qwen2.5-VL architecture; benchmarks incl. MIMIC-CXR, CheXpert Plus, IU-Xray report generation; avg multimodal VQA 61.8 (7B) / 66.6 (32B); grounding not mentioned — [HF card](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B). Inherits Qwen2.5-VL's box-output format but medical grounding unevaluated (inference). GPU ≈18 GB (7B), ≈70 GB (32B bf16).

**Items from prior knowledge, not re-fetched [K]**
- LLaVA-Med: "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day", C. Li et al. (Microsoft), NeurIPS 2023 Datasets & Benchmarks, arXiv:2306.00890; `microsoft/llava-med-v1.5-mistral-7b`; research-only license; ~7B; no boxes — [arXiv](https://arxiv.org/abs/2306.00890)
- LLaVA-Rad + CheXprompt: "A Clinically Accessible Small Multimodal Radiology Model and Evaluation Metric for Chest X-Ray Findings", J. Zambrano Chaves et al. (Microsoft), arXiv:2403.08002, published Nature Communications 2025; ~7B; `microsoft/llava-rad`; CheXprompt = GPT-4-based factual-error metric for CXR reports (it is a metric, not a model); no boxes — [arXiv](https://arxiv.org/abs/2403.08002)
- RadFM: "Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data", C. Wu et al. (SJTU/Shanghai AI Lab), arXiv:2308.02463 (Nature Communications 2025); ~14B; HF `chaoyi-wu/RadFM`; no boxes; ≈30 GB bf16 — [arXiv](https://arxiv.org/abs/2308.02463)
- BiomedCLIP: "BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs", S. Zhang et al. (Microsoft), arXiv:2303.00915 (NEJM AI 2025); `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, MIT, ~200M params, <2 GB GPU; zero-shot classification + patch-similarity maps only — [arXiv](https://arxiv.org/abs/2303.00915)
- CheXzero: "Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning", E. Tiu et al. (Harvard/Stanford), Nature Biomedical Engineering 2022, DOI 10.1038/s41551-022-00936-9; CLIP ViT-B/32 fine-tuned on MIMIC-CXR; weights via GitHub/Drive; zero-shot classification — [DOI](https://doi.org/10.1038/s41551-022-00936-9)
- MedCLIP: "MedCLIP: Contrastive Learning from Unpaired Medical Images and Text", Z. Wang et al. (UIUC), EMNLP 2022, arXiv:2210.10163; pip `medclip` — [arXiv](https://arxiv.org/abs/2210.10163)
- Med-Flamingo: M. Moor et al. (Stanford), arXiv:2307.15189, ML4H 2023; OpenFlamingo-9B based; no boxes — [arXiv](https://arxiv.org/abs/2307.15189)
- XrayGPT: O. Thawkar et al. (MBZUAI), arXiv:2306.07971 (BioNLP 2024); MedCLIP encoder + Vicuna-7B; no boxes — [arXiv](https://arxiv.org/abs/2306.07971)
- RaDialog: C. Pellegrini et al. (TUM), arXiv:2311.18681 (MIDL 2025); ~7B; no boxes — [arXiv](https://arxiv.org/abs/2311.18681)
- HuatuoGPT-Vision: "HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale", J. Chen et al. (CUHK-Shenzhen), arXiv:2406.19280; PubMedVision dataset; 7B/34B; Apache-2.0 [K] — [arXiv](https://arxiv.org/abs/2406.19280)
- Qwen2.5-VL Technical Report, Alibaba Qwen, arXiv:2502.13923; 3B/7B/32B/72B; native absolute-coordinate box/point output in JSON; Apache-2.0 (72B under Qwen license) — [arXiv](https://arxiv.org/abs/2502.13923)
- GPT-4V/4o and Gemini: closed APIs; uploading MIMIC-derived images to third-party APIs is restricted by the PhysioNet DUA except via approved setups (PhysioNet guidance on responsible LLM use, Azure OpenAI/Vertex with no-retention) [K — verify current wording].
- MedRAX (CXR reasoning agent wrapping tools incl. grounding, arXiv:2502.02673), ChestX-Reasoner (arXiv:2504.20930), Libra (temporal CXR RRG, arXiv:2411.19378), ChEX (interactive localization + region explanation, arXiv:2404.15770) [K].

### Inferences
- For a box-level reliability study, the practical open-weight grounded set is MAIRA-2 (7B), MedGemma 1.5 4B, RadVLM (credentialed), AnatomiX, and Qwen2.5-VL-7B (general-domain baseline); all fit on a single 24 GB GPU in bf16, except that 27B/32B+ need ≥48–80 GB or 4-bit quantization.
- MAIRA-2 and RadVLM trained on MS-CXR/MIMIC; evaluating them on MS-CXR is in-distribution and partly contaminated. PadChest-GR (Spain) or VinDr-CXR (Vietnam) are natural shift sets, but MAIRA-2 v2 already reports PadChest-GR numbers, so check its training splits.
- Text-only models (CheXagent-2, LLaVA-Rad, Lingshu) can still be scored for grounding by prompting for coordinates or via attention/Grad-CAM pointing game, but those numbers measure a different thing than native grounding.

### Gaps
- MAIRA-2 grounded result numbers (RadFact grounding precision/recall, box metrics) were not extracted from the full paper; only the abstract was fetched.
- Which dataset MedGemma 1.5's "35% IoU" is measured on (likely Chest ImaGenome/MS-CXR-style anatomy boxes) is unconfirmed.
- RadVLM base model, size, and HF repo name are not confirmed; MedVersa license not shown.
- No source found for an official head-to-head of GPT-4o/Gemini on MS-CXR grounding.

## Q2. Grounding datasets: size, annotation type, access

### Takeaway
MS-CXR (1,162 phrase–box pairs, MIMIC-based, PhysioNet credentialed) is the standard CXR phrase-grounding test set. VinDr-CXR (18k images, 22 box labels, credentialed) and ChestX-Det/Det10 (~3.5k NIH images, public GitHub) are the main detection-style shift sets. PadChest-GR (4,555 bilingual grounded-report studies, BIMCV on request) is the only multi-reader grounded-*report* set outside MIMIC. ReXGroundingCT and RadGenome-ChestCT are **CT**, not CXR.

### Cited Findings
- **MS-CXR** v1.1.0 (PhysioNet, 15 Nov 2024): 1,162 image–sentence pairs (1,026 reviewed/edited + 136 drawn from scratch); multiple boxes per sentence allowed; 8 findings (atelectasis, cardiomegaly, consolidation, edema, lung opacity, pleural effusion, pneumonia, pneumothorax); PhysioNet Credentialed Health Data License 1.5.0, needs credentialed account + CITI "Data or Specimens Only Research" + signed DUA; DOI 10.13026/9g2z-jg61; original paper Boecking et al., ECCV 2022 ("Making the Most of Text Semantics…", arXiv:2204.09817 [K]) — [PhysioNet](https://physionet.org/content/ms-cxr/)
- MS-CXR sample counts differ across papers after preprocessing (e.g., 890 samples/867 patients vs 1,153 triples) — compare only matched splits — [search summary of grounding papers](https://arxiv.org/html/2502.16585v1)
- **VinDr-CXR**: "VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations", H. Q. Nguyen et al. (VinBigData), arXiv:2012.15029, Scientific Data 2022; 18,000 PA CXRs (15,000 train / 3,000 test); 22 local labels with boxes + 6 global labels; train labeled by 3 radiologists independently, test by consensus of 5; PhysioNet credentialed access — [arXiv](https://arxiv.org/pdf/2012.15029); [PhysioNet](https://www.physionet.org/content/vindr-cxr/). (Do not confuse with VinDr-PCXR, pediatric — [PhysioNet](https://www.physionet.org/content/vindr-pcxr/))
- **ChestX-Det** (Deepwise AI Lab): from "A Structure-Aware Relation Network for Thoracic Diseases Detection and Segmentation", arXiv:2104.10326; subset of NIH ChestX-ray14; 13 categories; instance boxes + masks by 3 board-certified radiologists; 3,025 train / 553 test (≈3,578; paper also says 3,575 selected); released on GitHub Deepwise-AILab/ChestX-Det-Dataset (public) — [arXiv](https://arxiv.org/pdf/2104.10326)
- **ChestX-Det10**: box-level, 10 categories, ~3,500 NIH images, arXiv:2006.10550 — [HF papers mirror](https://huggingface-gw-kz.just-ai.com/papers/2006.10550)
- **PadChest-GR**: "PadChest-GR: A Bilingual Chest X-ray Dataset for Grounded Radiology Report Generation", D. C. Castro, A. Bustos, S. Bannur, S. L. Hyland, … A. Pertusa (14 authors; Microsoft + Univ. Alicante/Hospital San Juan); arXiv:2411.05085 (v2 Sep 2025); DOI 10.1056/AIdbp2401120 (NEJM AI); 4,555 studies (3,099 abnormal, 1,456 normal); 7,037 positive + 3,422 negative finding sentences; English+Spanish; each positive sentence has up to two independent reader box sets + labels for finding type, location, progression; download on request from BIMCV — [arXiv](https://arxiv.org/abs/2411.05085)
- **ReXGroundingCT** (CT, not CXR): "ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports", Baharoon, Luo, … Rajpurkar (25 authors), arXiv:2507.22030; 3,142 non-contrast chest CTs (from CT-RATE, Turkish reports machine-translated), 16,301 entities, 8,028 text→3D-segmentation pairs; 50 public val / 100 private test with leaderboard; ~79% focal findings; HF `rajpurkarlab/ReXGroundingCT` — [arXiv](https://arxiv.org/abs/2507.22030)
- **Chest ImaGenome** [K]: Wu et al., NeurIPS 2021 Datasets & Benchmarks; ~242k MIMIC-CXR frontal images with 29 anatomical-region boxes (silver, auto-generated; gold subset of 500 studies manually verified) and region–attribute–relation scene graphs; PhysioNet credentialed — [PhysioNet](https://physionet.org/content/chest-imagenome/)
- **RadGenome-ChestCT** [K] (CT): Zhang et al., arXiv:2404.16754; built on CT-RATE, region-guided reports + segmentation masks of 197 regions — [arXiv](https://arxiv.org/abs/2404.16754)
- **Other** [K]: GEMeX (grounded explainable CXR VQA on MIMIC-CXR/Chest ImaGenome, arXiv:2411.16778); REFLACX / EGD-CXR eye-gaze sets (PhysioNet); MIMIC-CXR itself (377k images, credentialed). "GRIT-style" CXR set: no dedicated CXR GRIT dataset found; general GRIT (Kosmos-2) is natural-image — UNVERIFIED for CXR.

### Inferences
- A shift protocol could train/calibrate on MIMIC-derived grounding (MS-CXR, Chest ImaGenome), then test on VinDr-CXR (Vietnam, different devices), ChestX-Det (NIH), and PadChest-GR (Spain, bilingual). The label spaces need mapping, e.g. to MS-CXR's 8 findings.
- Only ChestX-Det/Det10 can be downloaded without credentialing or a request, which makes them the quickest to start with. VinDr-CXR test boxes may not be public (one source says test labels withheld), so confirm on PhysioNet.

### Gaps
- Total box count for MS-CXR and PadChest-GR not stated on pages fetched.
- Whether VinDr-CXR test-set boxes are released on PhysioNet: conflicting/secondary information.
- Chest ImaGenome and RadGenome-ChestCT details are from prior knowledge, not re-fetched.

## Q3. How grounding is evaluated; who reports what

### Takeaway
CXR phrase grounding is mostly scored with box IoU: mIoU and Acc@IoU>0.5 (the MedRPG protocol) on MS-CXR. Grounded report generation adds RadFact (LLM entailment per sentence, plus box-level precision/recall on the grounded sentences; MAIRA-2) and CNR/contrast for heatmap models (BioViL). Pointing game and Grad-CAM IoU are used for post-hoc saliency. Segmentation sets (ReXGroundingCT) use Dice. Reported numbers are not comparable across papers because MS-CXR splits differ and some models trained on MS-CXR.

### Cited Findings
- MedGround-R1 reports Acc@IoU>0.5 and mIoU, following the MedRPG protocol, and retrains Qwen2.5-VL baselines — [arXiv](https://arxiv.org/pdf/2507.02994)
- On MRG-MS-CXR, classic grounding baselines (MedRPG, TransVG, RefTR, VGTR) stay below 45% mIoU — [arXiv](https://arxiv.org/pdf/2604.01915)
- Anatomical-grounding pretraining reports MS-CXR mIoU 61.2 after fine-tuning — [arXiv](https://arxiv.org/html/2502.16585v1)
- MAIRA-2 ≈57.8 mIoU on MS-CXR, flagged as contaminated, from a secondary source — [emergentmind](https://www.emergentmind.com/topics/maira-2)
- MAIRA-2 introduces RadFact for grounded report evaluation — [arXiv](https://arxiv.org/abs/2406.04449)
- MedGemma 1.5 reports CXR localization IoU (+35%) — [arXiv](https://arxiv.org/abs/2604.05081)
- AnatomiX reports anatomy grounding, phrase grounding, grounded diagnosis, and grounded captioning — [arXiv](https://arxiv.org/abs/2601.03191)
- RadVLM reports visual grounding SOTA — [arXiv](https://arxiv.org/html/2502.03333v1)
- ReXGroundingCT uses a public leaderboard on a private 100-case test set — [arXiv](https://arxiv.org/abs/2507.22030)
- [K] Other metrics in use: BioViL/BioViL-T report CNR on MS-CXR for heatmaps (Boecking 2022, arXiv:2204.09817); RadGraph-F1 and GREEN (arXiv:2405.03595) / RaTEScore / CheXprompt score report text, not location; the pointing game (Zhang et al., 2016) is standard for saliency.

### Inferences
- For an "evidence-grounded reliability" paper, a defensible metric set is box mIoU, Acc@IoU≥0.3/0.5, and pointing-game hit rate (for text-only models via Grad-CAM). Add text–box consistency (RadFact-style: is the sentence entailed AND is its box correct?) and calibration of grounding confidence under shift.
- Report MS-CXR split provenance and training overlap for every model.

### Gaps
- No primary-source head-to-head for GPT-4o/Gemini on MS-CXR or VinDr box grounding was found in this session.
- RadFact's exact box-metric definitions were not extracted from the MAIRA-2 full text.
