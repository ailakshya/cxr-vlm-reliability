# Reliability evaluation of medical (CXR) VLMs: faithfulness metrics, calibration/abstention, benchmarks, gap analysis (2023–2026)

Verification note: arXiv IDs marked "verified" were checked against the arXiv export API on 2026-10-29 (title + date match). Items marked UNVERIFIED were not confirmed with a tool. Abstract-level facts come from the arXiv abstracts; numbers not in abstracts come from search snippets and are flagged.

## Q1. Report-quality and faithfulness metrics — which ones check claims against image evidence?

### Takeaway
Nearly every CXR report metric (RadGraph F1, CheXbert F1, RadCliQ, GREEN, RaTEScore, FineRadScore, CheXprompt, RadFact's logical part) compares the generated report with a **reference report**, not with the image. Only RadFact's grounding component (MAIRA-2) checks spatial grounding (bounding boxes), and it still scores boxes against reference annotations. Reference-free, image-evidence verification of each claim is basically missing, and that gap is the opening for an "evidence-grounded" paper.

### Cited Findings
- **CheXbert** (2020): a BERT labeler for 14 CheXpert observations. "CheXbert F1" (micro/macro on 14 or 5 labels) compares labels pulled from the generated report against labels from the reference report. Reference-based, label-level only. arXiv 2004.09167 (verified) — [arXiv](https://arxiv.org/abs/2004.09167)
- **RadGraph** (2021): entity/relation extraction from reports. RadGraph F1 = overlap of entities and relations between generated and reference reports. Reference-based. arXiv 2106.14463 (verified) — [arXiv](https://arxiv.org/abs/2106.14463)
- **RadCliQ** (Yu et al., "Evaluating progress in automatic chest X-ray radiology report generation", Patterns 2023): a learned combination of BLEU, BERTScore, CheXbert similarity and RadGraph F1, fitted to radiologist error counts. Reference-based. DOI 10.1016/j.patter.2023.100802 is from memory, so treat it as UNVERIFIED. (One candidate arXiv ID I tried, 2209.12177, turned out to be a different paper.) — [DOI](https://doi.org/10.1016/j.patter.2023.100802)
- **GREEN** (2024): an LLM-based metric that counts clinically significant and insignificant errors (6 categories) in the candidate against the reference, with explanations. Reference-based. arXiv 2405.03595 (verified) — [arXiv](https://arxiv.org/abs/2405.03595)
- **RaTEScore** (2024): an entity-aware metric (medical NER + synonym-robust embeddings, negation-aware). Reference-based. arXiv 2406.16845 (verified) — [arXiv](https://arxiv.org/abs/2406.16845)
- **FineRadScore** (2024): line-by-line LLM evaluation that produces corrections plus severity scores. Reference-based. arXiv 2405.20613 (verified) — [arXiv](https://arxiv.org/abs/2405.20613)
- **CheXprompt** (Microsoft, in the LLaVA-Rad paper "Towards a clinically accessible radiology foundation model…with automated evaluation"): GPT-4-based error counting against the reference, in the RadCliQ-style taxonomy. arXiv 2403.08002 (verified) — [arXiv](https://arxiv.org/abs/2403.08002)
- **RadFact** (Microsoft, MAIRA-2 paper): uses LLM logical inference to score "report correctness and completeness at the level of individual sentences, while supporting the new task of grounded reporting". Logical precision/recall is entailment of each generated sentence against reference sentences, and vice versa. Grounding precision/recall is a binary per-sentence spatial check of predicted boxes against reference boxes. arXiv 2406.04449 (verified) — [arXiv](https://arxiv.org/abs/2406.04449). A snippet reports MAIRA-2 7B at roughly 68 grounding precision and 92 grounding recall, with logical precision around 74 (snippet-level, check against Table) — [MS Research PDF](https://www.microsoft.com/en-us/research/uploads/prodnew/2024/06/2406.04449v1.pdf)
- **ReXrank** (Harvard/Gradient/Duke, Nov 2024): a public leaderboard with 8 metrics (lexical plus BERTScore, SembScore, RadGraph-F1, 1/RadCliQ, RaTEScore, GREEN, FineRadScore per the paper). It uses MIMIC-CXR, IU-Xray, CheXpert Plus and the private ReXGradient set of 10,000 studies, with separate tracks for findings-only and findings+impression. Because it covers several datasets it is implicitly multi-site, but it evaluates accuracy only: no calibration and no grounding. arXiv 2411.15122 (verified) — [arXiv](https://arxiv.org/abs/2411.15122); [rexrank.ai](https://rexrank.ai)
- **FactCheXcker** (CVPR 2025): a pipeline that mitigates *measurement* hallucinations (e.g., ETT position) through query-code-update. It was evaluated on MIMIC-CXR with 11 RRG models and reports an average 135% improvement in measurement MAE. It does check the image, by re-measuring with a dedicated detector, but only for measurements. arXiv 2411.18672 (verified) — [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Heiman_FactCheXcker_Mitigating_Measurement_Hallucinations_in_Chest_X-ray_Report_Generation_Models_CVPR_2025_paper.pdf)
- **ReXTrust** (Dec 2024): white-box, finding-level hallucination detection from LVLM hidden states. AUROC 0.8751 on all findings and 0.8963 on clinically significant findings (MIMIC-CXR subset). This is reference-free at inference, but it is a model-internal signal and does not check image evidence explicitly. arXiv 2412.15264 (verified) — [arXiv](https://arxiv.org/abs/2412.15264)
- **Semantic-consistency UQ for RRG** (Dec 2024): black-box report- and sentence-level uncertainty from sampling consistency. Abstaining on the 20% most uncertain reports improves factuality by 10% (Radialog, MIMIC-CXR). arXiv 2412.04606 (verified) — [arXiv](https://arxiv.org/abs/2412.04606)
- **HalluCXR** (May 2026): 6 VLMs, 856 MIMIC-CXR images, 3 query types, 15,408 evaluations. It uses an 8-category hallucination taxonomy with a detection pipeline validated on 250 human annotations (auto F1 0.959). 61.9–82.3% of outputs contain hallucinations, and response length predicts hallucination (AUC up to 0.908). It is single-source (MIMIC only). arXiv 2605.20469 (verified) — [arXiv](https://arxiv.org/abs/2605.20469). A search snippet says it also reports ECE (Qwen2-VL 0.725, BiomedCLIP 0.181, Gemini 3 Flash 0.113); this is snippet-level, so verify in the full text — [HTML](https://arxiv.org/html/2605.20469v1)

### Inferences
- Grounding checks fall into three tiers: (a) none, which is reference-text-only (RadGraph, CheXbert, RadCliQ, GREEN, RaTEScore, FineRadScore, CheXprompt, RadFact-logical); (b) spatial agreement with reference boxes (RadFact-grounding); (c) independent image re-verification for a narrow claim type (FactCheXcker measurements). No metric I found checks every positive finding claim against independent image evidence, such as a detector/segmenter or expert bounding-box sets like VinDr-CXR/ChestX-Det.
- All the reference-based metrics break when the reference reports change style across sites (e.g., PadChest Spanish reports, ReXGradient US community reports). That is a confound for any cross-site evaluation built on them, and an argument for label- or box-based evidence checks.

### Gaps
- I could not confirm whether ReXrank added calibration or grounding tracks after 2024 (the live leaderboard was not fetched).
- ReXGradient-160K and CheXpert Plus arXiv IDs were not tool-verified (UNVERIFIED).

## Q2. Benchmarks: aspects covered, models, distribution shift

### Takeaway
Trustworthiness and hallucination benchmarks are crowded, but they are mostly multi-modality VQA built from public VQA sets (SLAKE, VQA-RAD, PathVQA, MIMIC-based). Only a few measure calibration, and almost none combine site shift with grounding. The 2026 wave (MVC-Bench, MedQ-Deg, HalluCXR, MC-CXR) is starting to add calibration and degradation axes, so the space is filling quickly.

### Cited Findings (one line per benchmark)
- **CARES** (NeurIPS 2024 D&B): trustfulness, fairness, safety, privacy and robustness. About 41K QA pairs, 16 modalities, 27 regions. Its robustness axis is mostly OOD detection and abstention on out-of-scope inputs, not site shift. Calibration is touched under trustfulness via uncertainty/abstention. arXiv 2406.06007 (verified) — [arXiv](https://arxiv.org/abs/2406.06007); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/fde7f40f8ced5735006810534dc66b33-Paper-Datasets_and_Benchmarks_Track.pdf)
- **ProbMed** ("Worse than Random?"): probing VQA with adversarial negated or hallucinated attributes. It shows LMMs (including GPT-4V and Gemini) at or below random on fine-grained diagnostic probes. Covers hallucination and robustness to adversarial questions. arXiv 2405.20421 (verified) — [arXiv](https://arxiv.org/abs/2405.20421)
- **MedVH**: hallucination evaluation of LVLMs in the medical context (false-confidence and wrong-premise tasks). arXiv 2407.02730 (verified) — [arXiv](https://arxiv.org/abs/2407.02730)
- **Med-HallMark / MediHall Score** ("Detecting and Evaluating Medical Hallucinations in LVLMs"): a hallucination benchmark plus detector (MediHallDetector). arXiv 2406.10185 (verified) — [arXiv](https://arxiv.org/abs/2406.10185)
- **MedHallBench**: hallucination benchmark for medical LLMs/LVLMs. arXiv 2412.18947 (verified) — [arXiv](https://arxiv.org/abs/2412.18947)
- **MedHEval** (Mar 2025): hallucination benchmark plus mitigation comparison for Med-LVLMs. arXiv 2503.02157 (verified) — [arXiv](https://arxiv.org/abs/2503.02157)
- **RadVUQA** ("Beyond the Hype"): five areas, namely anatomy, multimodal comprehension, quantitative/spatial reasoning, physiological knowledge, and robustness to unharmonized and synthetic data. CT/MRI rather than CXR (per v2 abstract). arXiv 2408.08704 (verified via API) — [arXiv](https://arxiv.org/abs/2408.08704)
- **GMAI-MMBench**: broad capability coverage across many modalities and tasks. Accuracy only, no calibration or shift. arXiv 2408.03361 (verified) — [arXiv](https://arxiv.org/abs/2408.03361)
- **OmniMedVQA** (CVPR 2024): large multi-modality VQA built from about 73 datasets. Accuracy only. arXiv 2402.09181 (verified) — [arXiv](https://arxiv.org/abs/2402.09181)
- **MultiMedEval**: a toolkit that runs many medical VLM tasks (VQA, RRG with CheXbert/RadGraph metrics, classification). Accuracy and quality only. arXiv 2402.09262 (verified) — [arXiv](https://arxiv.org/abs/2402.09262)
- **ReXVQA** (Jun 2025): large-scale CXR VQA for "generalist CXR understanding". Accuracy-centric. arXiv 2506.04353 (verified) — [arXiv](https://arxiv.org/abs/2506.04353)
- **CheXbench** (in the CheXagent paper): 8 CXR task groups (view, disease, VQA, report, etc.). Accuracy. arXiv 2401.12208 (verified) — [arXiv](https://arxiv.org/abs/2401.12208)
- **RadBench** (in the RadFM paper): 2D/3D radiology generalist evaluation. Accuracy. arXiv 2308.02463 (verified) — [arXiv](https://arxiv.org/abs/2308.02463)
- **CheXGenBench** (May 2025): evaluates *synthetic CXR image generators* on fidelity, privacy and utility. **Not a VLM reliability benchmark**, so it is irrelevant except as a source of shifted or synthetic images. arXiv 2505.10496 (verified) — [arXiv](https://arxiv.org/abs/2505.10496)
- **SURE-VQA** (Nov 2024): a framework for robustness evaluation in medical VQA under realistic shifts, which argues that synthetic-only shifts are insufficient. arXiv 2411.19688 (verified) — [arXiv](https://arxiv.org/abs/2411.19688)
- **MediMeta-C / RobustMedCLIP** (May 2025): a corruption benchmark (with MedMNIST-C) testing 5 MVLMs over 5 modalities. Shows "severe degradation under corruption". Classification only, CLIP-style models, no calibration. arXiv 2505.15425 (verified) — [arXiv](https://arxiv.org/abs/2505.15425)
- **VLM artefact robustness** (npj Digital Medicine 2025): CXR, brain MRI and fundus with injected artefacts. Accuracy drops with weak artefacts and strong artefacts are rarely flagged. Classification-style — [Nature](https://www.nature.com/articles/s41746-025-02108-w); [medRxiv](https://www.medrxiv.org/content/10.1101/2025.05.13.25327495v1)
- **MedQ-Deg** (Mar 2026): 18 degradation types × 3 radiologist-calibrated severities, 7 modalities, 24,894 QA, 40 MLLMs. Introduces a "Calibration Shift" metric (the gap between perceived confidence and performance under degradation). VQA, not report generation, and no grounding. arXiv 2603.07769 (verified) — [arXiv](https://arxiv.org/abs/2603.07769)
- **MVC-Bench** (Aug 2026): a calibration-centric benchmark for medical *image classification* with VLMs. 8 backbones, fundus/histopathology/CXR, in-domain plus domain shift, post-hoc vs. train-time calibration, 6 prompt-tuning methods, more than 1,638 experiments, ECE as the primary metric. CLIP-style classification, not generative reports. arXiv 2608.27004 (verified) — [arXiv](https://arxiv.org/abs/2608.27004)
- **MC-CXR** (Aug 2026): context-induced disruption. 240 radiologist-curated CXR cases expanded to 2,522 instances, 10 VLMs. Answer switch rates are 45.6–78.1% under misleading text and 35.7–61.7% under misleading visual context (snippet numbers). This is a context/prompt shift, not a site shift. arXiv 2608.24118 (verified) — [arXiv](https://arxiv.org/abs/2608.24118)
- **HalluCXR** (May 2026): see Q1. CXR hallucination taxonomy plus some ECE, MIMIC-only. arXiv 2605.20469 (verified) — [arXiv](https://arxiv.org/abs/2605.20469)
- **Designing a Robust RRG System** (Nov 2024): a system-design paper on robustness for RRG. arXiv 2411.01153 (verified; content not read) — [arXiv](https://arxiv.org/pdf/2411.01153)

### Inferences
- **Site shift is the least-covered axis.** Benchmarks either pool many public datasets (OmniMedVQA, GMAI) without framing site as a shift variable, or use synthetic corruptions (MediMeta-C, MedQ-Deg). ReXrank spans 4 sources but does not analyse shift. MVC-Bench and 2604.08941 (Q3) are the only ones I found that frame in-domain vs. domain-shift calibration explicitly with CXR.
- **Grounding is absent from every benchmark listed.** None scores whether a VLM's claimed finding is localised in the correct image region.

### Gaps
- I did not read the full texts of CARES, ProbMed and GMAI to confirm the exact model lists. Commonly evaluated models (LLaVA-Med, Med-Flamingo, MedVInT, RadFM, CheXagent, GPT-4V) are from memory and UNVERIFIED per benchmark.
- UR²-MLLM (arXiv 2608.22217, "Uncertainty-aware Revisit Reasoning… for RRG") showed up in search but was not verified or read (UNVERIFIED).

## Q3. Calibration, uncertainty and selective prediction for medical VLMs — any on CXR?

### Takeaway
There is very active 2026 work on verbalized-confidence calibration, conformal/certified selective prediction and entropy-based abstention for medical VQA. A few studies use CXR, including one with explicit MIMIC→PadChest cross-site shift (2604.08941). Conformal factuality for generated text exists (general LVLMs, and Conformal Alignment with an RRG use case), but I found nothing that combines it with image-grounded claim checking on CXR under site shift.

### Cited Findings
- **Conformal Factuality** (Mohri & Hashimoto 2024): splits LM output into claims and filters them with conformal guarantees on factuality. Text-only LMs. arXiv 2402.10978 (verified) — [arXiv](https://arxiv.org/abs/2402.10978)
- **Conformal Alignment** (Gui, Jin, Ren 2024): selects outputs guaranteed to be aligned, e.g., at least 99% of selected reports match expert reports. **Radiology report generation is a main use case.** arXiv 2405.10301 (verified) — [arXiv](https://arxiv.org/abs/2405.10301)
- **ConfLVLM** ("Towards Statistical Factuality Guarantee for LVLMs", Feb 2025): treats each generated detail as a hypothesis and filters claims with finite-sample, distribution-free guarantees. General LVLMs. The search snippet did not confirm whether it has a medical or radiology experiment, so verify that. arXiv 2502.20560 (verified) — [arXiv](https://arxiv.org/abs/2502.20560)
- **IntroConformal** (Sep 2026): conformal factuality for LVLMs using introspective signals. Experiments are on scene understanding, captioning and documents, with no radiology shown in the snippet. arXiv 2609.01375 (verified) — [arXiv](https://arxiv.org/abs/2609.01375)
- **SCoRE** (Mar 2026): selective conformal risk control with e-values, with an extension to distribution shift. One application scores LLM radiology reports with 14-dim condition vectors (CheXpert-style). arXiv 2603.24704 (verified) — [arXiv](https://arxiv.org/abs/2603.24704)
- **CONRep** (J. Imaging Informatics in Medicine / Springer, 2026): conformal prediction for VLM impression drafting on ChestX-Det10 (label-based) and Open-I (sentence-based). "Certain" outputs agree significantly more with GT. DOI 10.1007/s10278-026-02179-5 (from the URL; content via snippet) — [Springer](https://link.springer.com/article/10.1007/s10278-026-02179-5)
- **Semantic entropy** (Farquhar et al., Nature 2024) is the generic hallucination-detection basis. DOI 10.1038/s41586-024-07421-0 is from memory, so treat it as UNVERIFIED. The CXR-specific analogue is 2412.04606 (see Q1).
- **"Predictive Entropy as a Joint Screen for Error and Paraphrase Instability in Medical VLMs"** (Apr 2026; v1 title was "…Links Calibration and Paraphrase Sensitivity…"): MedGemma-4B-IT on MIMIC-CXR (ID, n=196) and **PadChest (OOD, n=861)**, binary presence questions. Entropy predicts paraphrase flips with AUROC 0.82. At 80% coverage, error is 3.3% and flip rate 5.7%. Under cross-site shift, at a 5% risk target, a single forward pass answers 88.5% of cases, matching MC Dropout and beating a 5-seed ensemble. Replicates on LLaVA-Rad-7B. **This is the closest existing work on CXR VLM + site shift + selective prediction.** It is small-n, binary VQA only, and has no grounding. arXiv 2604.08941 (verified) — [arXiv](https://arxiv.org/abs/2604.08941)
- **"Adapt or Calibrate? How Local Labels Buy Certified Coverage in Medical VQA"** (Jun 2026; an earlier version was titled "Calibrated Triage, Not Autonomy…"): 9 confidence estimators × 5 open-weight VLMs × 3 med-VQA sets, using Learn-then-Test certification. The best pooled certificate covers 11.5% of SLAKE at 20% error and 0% on PATH-VQA. Adapting a probe with 500 local labels raises coverage from 6.8% to 37.6%. arXiv 2606.15910 (verified) — [arXiv](https://arxiv.org/abs/2606.15910)
- **"Just how sure are you?"** (Jun 2026): fine-tuning to improve verbalized uncertainty (MedGemma, Qwen2-VL) on medical VQA. Reports the best average ECE, Brier and AUROC on 3 benchmarks. arXiv 2606.27023 (verified) — [arXiv](https://arxiv.org/abs/2606.27023)
- **DualRead / CCG-AUC** (Sep 2026): separates capability from confidence for GRPO-trained medical VLMs (pre-answer solvability plus post-answer visual-support assessment). It introduces **Counterfactual Confidence Grounding AUC**, which tests whether confidence drops when image substitution makes the model wrong. Evaluated on ID and OOD med-VQA. **This is the closest to an "evidence-grounded confidence" idea**, but it works through image substitution, not region-level evidence, and on VQA rather than CXR reports. arXiv 2609.06419 (verified) — [arXiv](https://arxiv.org/abs/2609.06419)
- **"Confident but Unreliable"** (Aug 2026): verbalized 0–100 confidence on brain MRI VLMs. ECE 0.27–0.40, and 33–46% of answered items are high-confidence errors. Argues for reporting abstention alongside accuracy (snippet numbers). arXiv 2608.02790 (verified) — [arXiv](https://arxiv.org/abs/2608.02790)
- **CALM-VLM**: temperature scaling plus selective abstention for a 3D brain-MRI VLM — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13104997/)
- **Reducing Unnecessary Abstention in VL models** (Findings ACL 2024): general domain — [ACL Anthology](https://aclanthology.org/2024.findings-acl.767.pdf)
- A search snippet mentioned "a CXR study on MedGemma-4B benchmarking five UQ methods" (entropy-based). It may be the same as 2604.08941. Not separately identified (UNVERIFIED).

### Inferences
- The three ingredients (calibration/selective prediction on CXR VLMs, site shift, conformal claim filtering) each exist separately as of Oct 2026. Their union with an **image-evidence grounding check** does not exist in what I found.
- Most of this calibration work is on closed/binary VQA, where confidence is well defined. Calibration of *free-text reports* at the claim level (the RRG setting) under site shift is thinner. Only 2412.04606, Conformal Alignment, CONRep and SCoRE touch RRG, and none does cross-site evaluation with grounding.

### Gaps
- I did not check whether ConfLVLM includes a medical-report experiment.
- I could not determine whether any 2026 paper applies conformal factuality to MAIRA-2 grounded outputs.

## Q4. Gap analysis: (grounding evidence check + multi-site/corruption shift + calibration/abstention) on CXR VLMs

### Takeaway
No paper I found does all three. The closest five each cover at most two axes. The window is open but narrowing quickly: 2026 has brought roughly one new adjacent preprint per month (MedQ-Deg in March, 2604.08941 in April, HalluCXR in May, 2606.x in June, MVC-Bench and MC-CXR in August, DualRead and IntroConformal in September). Scoop risk is **moderate-to-high** for a pure "benchmark that measures calibration under shift". It is **lower** if the contribution centres on claim-level *image-evidence* verification (boxes/masks from VinDr-CXR, ChestX-Det, MS-CXR, or REFLACX-type sets) used both as a faithfulness metric and as an abstention signal across sites.

### Closest papers and how a new paper would differ
1. **2604.08941 (Predictive entropy, MIMIC→PadChest, MedGemma/LLaVA-Rad)** has calibration/selective prediction and cross-site shift, but no grounding, binary VQA only, and n≈1k. A new paper would add claim-level evidence checks, more sites (CheXpert, VinDr-CXR, PadChest, ReXGradient), corruptions, and generative reports. — [arXiv](https://arxiv.org/abs/2604.08941)
2. **HalluCXR (2605.20469)** has hallucination taxonomy and some ECE on CXR, but is MIMIC-only, with no shift and no localisation evidence. A new paper would add site/corruption shift and box-grounded verification. — [arXiv](https://arxiv.org/abs/2605.20469)
3. **MVC-Bench (2608.27004)** has calibration plus domain shift including CXR, but is CLIP-style classification, not generative VLMs, with no hallucination or grounding. A new paper differs by covering generative VLM outputs and evidence checks. — [arXiv](https://arxiv.org/abs/2608.27004)
4. **MAIRA-2 / RadFact (2406.04449)** has grounded RRG and a grounding metric, but is reference-dependent, with no calibration/abstention and no shift study. A new paper would use RadFact-style grounding as one evidence signal and test it under shift with selective prediction. — [arXiv](https://arxiv.org/abs/2406.04449)
5. **DualRead / CCG-AUC (2609.06419)** has confidence that reflects visual support, plus OOD VQA. But its notion of grounding is counterfactual image substitution (not region evidence), and it is not CXR-report specific. A new paper differs with region-level evidence and CXR multi-site evaluation. — [arXiv](https://arxiv.org/abs/2609.06419)
- Honourable mentions: MedQ-Deg (degradation plus Calibration Shift, VQA, no grounding) — [arXiv](https://arxiv.org/abs/2603.07769); semantic-consistency UQ for RRG (abstention, MIMIC-only) — [arXiv](https://arxiv.org/abs/2412.04606); Conformal Alignment (RRG selection guarantee, no shift or grounding) — [arXiv](https://arxiv.org/abs/2405.10301).

### Inferences
- A defensible contribution statement would read roughly like this: "the first evaluation of CXR VLMs where each positive finding claim is checked against independent image evidence (expert boxes or detector agreement), and where that evidence-support rate is used as (i) a reference-free faithfulness metric robust to cross-site report-style differences and (ii) an abstention signal compared against entropy/verbalized confidence, across ≥3 sites plus graded corruptions, with calibration (ECE/AURC) reported per site."
- Shift-robust conformal methods (SCoRE claims a shift extension) make "do conformal guarantees survive site shift for CXR reports?" a natural sub-question, which no one has answered for CXR.
- To lower scoop risk: post an arXiv preprint early, and lean on the evidence-check part rather than generic calibration, which MVC-Bench, MedQ-Deg and 2604.08941 already occupy.

### Gaps
- Search coverage of Oct 2026 preprints is incomplete. There may be unindexed works, for example MICCAI 2026 papers or workshop papers such as UNSURE 2026.

## Q5. Venues and deadlines (2026–2027)

### Takeaway
The nearest realistic targets are MIDL 2027 (full papers due around early Dec 2026, if it follows the 2026 pattern) and MICCAI 2027 (due around late Feb or early Mar 2027; conference Sep 27–Oct 1 2027, Auckland). NeurIPS 2027 D&B is the fit if the paper is a benchmark. Journals (MedIA, TMI, JBHI, CIBM, npj Digit Med) accept rolling submissions.

### Cited Findings
- MIDL 2026 dates: full papers had abstract registration Dec 1–3, 2025 and submission Dec 5–10, 2025. The short-paper deadline was Apr 15, 2026. The conference ran Jul 8–10, 2026, Taipei — [MIDL 2026 CFP](https://2026.midl.io/call-for-papers.html)
- MICCAI past deadlines: 2024 paper deadline Mar 7, 2024. 2026: abstract Feb 12, 2026 and paper Feb 26, 2026. MICCAI 2027: deadline TBA; conference Sep 27–Oct 1, 2027, Auckland, NZ — [mldeadlines](https://mldeadlines.com/conference/miccai/)

### Inferences
- If the patterns hold, MIDL 2027 full papers are due around early Dec 2026. That is roughly 5 weeks from today, and too tight unless experiments already exist. MICCAI 2027 is likely due around late Feb 2027. The MICCAI 2027 UNSURE workshop (uncertainty) typically has a deadline in late June or July, which is a fallback.
- NeurIPS D&B deadlines have historically fallen in May. If NeurIPS 2027 follows that, the deadline is around May 2027 (UNVERIFIED for 2027).
- Medical Image Analysis, IEEE TMI, IEEE JBHI, Computers in Biology and Medicine and npj Digital Medicine have no deadlines (rolling). Turnaround is typically 3–9 months. CIBM and JBHI are the faster and more accessible options for an M.Tech project, while MedIA and TMI expect methodological novelty beyond evaluation. These are general field norms, not sourced here.

### Gaps
- 2027 official dates for MIDL, MICCAI and NeurIPS D&B were not announced or found as of this search.
