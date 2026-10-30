# Distribution Shift and Robustness for Chest X-ray AI (with emphasis on VLMs), 2019–2026

Verification note: items marked **[fetched]** were confirmed this session through a search result or a fetched abstract/page. Items marked **[known]** come from well-established papers I know (canonical arXiv ID or DOI given) but did not re-open this session; the numbers should be spot-checked before they go into the paper. **UNVERIFIED** means I could not confirm the item exists or that the detail is correct.

## Q1. Cross-hospital/dataset shift: which datasets, sizes, labels, access rules, and are free-text reports available?

### Takeaway
Only a few public CXR datasets ship free-text reports that can be used to evaluate report generation: MIMIC-CXR (PhysioNet credentialed), CheXpert Plus (Stanford AIMI registration), IU-Xray/Open-i (open), PadChest (Spanish reports, BIMCV registration), PadChest-GR (bilingual, grounded) and ReXGradient-160K (multi-site US). Most other "external" sets (NIH14, VinDr-CXR, BRAX, SIIM, RSNA, Shenzhen/Montgomery) have only labels or boxes. That limits report-level shift studies to roughly MIMIC → {CheXpert Plus, IU-Xray, PadChest, ReXGradient}, with label-based or grounding-based evaluation on the rest.

### Cited Findings
**Datasets with reports**
- MIMIC-CXR v2.0.0 (Beth Israel Deaconess, Boston): 377,110 images, 227,835 studies, 65,379 patients, with free-text reports. Access is PhysioNet *credentialed* (CITI human-subjects training plus a signed DUA). MIMIC-CXR-JPG adds 14 CheXpert/NegBio labels. [known] — [PhysioNet MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/); [Johnson et al., Sci Data 2019](https://doi.org/10.1038/s41597-019-0322-0)
- CheXpert (Stanford): 224,316 radiographs from 65,240 patients. It has 14 observations labeled by a rule-based labeler with *uncertain* labels, and a 200/500-study radiologist-labeled val/test set. The original release had no reports. Access is a Stanford AIMI research-use registration. [known] — [Irvin et al., arXiv 1901.07031](https://arxiv.org/abs/1901.07031)
- CheXpert Plus (2024, arXiv 2405.19538): 187,711 distinct reports, one per study, aligned to DICOM images. It has 36M text tokens (13M of them in impressions), patient demographics, pathology labels and RadGraph annotations. Almost 1M PHI spans were de-identified. The authors call it "the largest text dataset publicly released in radiology." Authors: Chambon, Delbrouck, …, Langlotz. [fetched] — [arXiv 2405.19538](https://arxiv.org/abs/2405.19538)
- IU-Xray / Open-i (Indiana University): about 3,955 reports and about 7,470 images, openly downloadable with no credential. It is commonly used as the *external* report-generation test set. [known] — [Demner-Fushman et al., JAMIA 2016](https://doi.org/10.1093/jamia/ocv080)
- PadChest (Hospital San Juan, Alicante, Spain, 2009–2017): about 160,868 images from about 67k patients, with **Spanish** reports. It has 174 radiographic findings, 19 differential diagnoses and 104 anatomic locations. About 27% were labeled manually by physicians; the rest were labeled by an RNN trained on those. Access is a BIMCV registration form. [known] — [Bustos et al., arXiv 1901.07441](https://arxiv.org/abs/1901.07441)
- PadChest-GR (2024): a bilingual (Spanish/English) dataset for *grounded* radiology report generation, with sentence-level findings linked to bounding boxes. [fetched: existence/title] — [arXiv 2411.05085](https://arxiv.org/abs/2411.05085)
- ReXGradient-160K (2025): 160,000 CXR studies paired with free-text reports, from multiple US institutions (Gradient Health / Rajpurkar lab). Splits are 140k train, 10k val and 10k public test, plus a 10k *private* test set reserved for the ReXrank leaderboard. [fetched via secondary summaries] — [CatalyzeX listing](https://www.catalyzex.com/paper/rexgradient-160k-a-large-scale-publicly); [DotMed news](https://www.dotmed.com/legal/print/story.html?nid=64734). The arXiv ID (believed to be 2505.00228) is **UNVERIFIED**.

**Label-only / annotation-only datasets (no public reports)**
- NIH ChestX-ray14: 112,120 frontal images from 30,805 patients. It has 14 NLP-mined labels plus about 1,000 bounding boxes. Download is open with no reports. [known] — [Wang et al., arXiv 1705.02315](https://arxiv.org/abs/1705.02315)
- VinDr-CXR (Vietnam, two hospitals): 18,000 PA images. The 15,000 training images were each read by 3 radiologists; the 3,000 test images carry a 5-radiologist consensus. Labels are 22 local findings with boxes plus 6 global diagnoses. Access is PhysioNet credentialed. No reports. [known] — [Nguyen et al., arXiv 2012.15029](https://arxiv.org/abs/2012.15029)
- BRAX (Hospital Israelita Albert Einstein, Brazil): about 40,967 images, 24,959 studies, 19,351 patients. It has 14 CheXpert-style labels mined from Portuguese reports. Access is PhysioNet credentialed. Reports are not distributed. [known] — [Reis et al., Sci Data 2022](https://doi.org/10.1038/s41597-022-01608-8)
- CANDID-PTX (Dunedin Hospital, New Zealand): 19,237 CXRs with pneumothorax segmentations, acute rib fracture and chest tube annotations. Access requires an ethics course plus a DUA. Whether free-text reports are included is **UNVERIFIED**. [known] — [Feng et al., Radiology: AI 2021](https://doi.org/10.1148/ryai.2021210136)
- SIIM-ACR Pneumothorax (Kaggle 2019): about 12k DICOMs with pixel masks, drawn from NIH14 images. No reports. RSNA Pneumonia Detection (Kaggle 2018): about 26.7k NIH14 images with opacity boxes. No reports. [known] — [Shih et al., Radiology: AI 2019](https://doi.org/10.1148/ryai.2019180041)
- Shenzhen (China; 662 images, 336 TB) and Montgomery County (USA; 138 images, 58 TB) are openly released by NLM, with TB labels and lung masks only. [known] — [Jaeger et al., QIMS 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4256233/)

**Indian datasets**
- IN-CXR (ICMR–National Institute for Research in Tuberculosis): PA DICOM images from the 2019–21 national TB Prevalence Survey, grouped as normal/abnormal. No reports. [fetched] — [NIRT IN-CXR](https://nirt.res.in/incxr)
- A Mendeley COVID-era CXR set from 3 hospitals in Uttar Pradesh and Rajasthan. [fetched] — [Mendeley 4n66brtp4j](https://data.mendeley.com/datasets/4n66brtp4j)
- **IND1** is a private Indian in/outpatient dataset with reports, used by Google DeepMind for Flamingo-CXR and Med-Gemini. It is not public. The Med-Gemini authors note that RadGraph F1 was not computed on IND1 because RadGraph is trained on MIMIC-style reports. [fetched] — [Flamingo-CXR arXiv 2311.18260](https://arxiv.org/abs/2311.18260); [Med-Gemini arXiv 2405.03162](https://arxiv.org/abs/2405.03162)
- Large multi-site Indian studies such as the "Autonomous AI … Multi-Site Study in the Indian Healthcare System" (5,003,742 CXRs) use proprietary data. [fetched] — [arXiv 2504.00022](https://arxiv.org/abs/2504.00022)

### Inferences
- A practical report-generation shift protocol is: train or adopt models on MIMIC-CXR, then test on CheXpert Plus (US, different hospital), IU-Xray (US, small, open), PadChest/PadChest-GR (Spain, language/translation shift) and ReXGradient (multi-site US). VinDr-CXR, BRAX and CANDID-PTX can be used for *finding-level* or *grounding-level* checks against boxes and labels, even without reference reports.
- For an India-based M.Tech project, there is no public Indian CXR dataset with reports. IN-CXR supports only normal/abnormal or TB-style checks.

### Gaps
- Exact current licence terms (CheXpert Plus DUA wording, ReXGradient hosting and gating) were not fetched.
- CANDID-PTX report availability is unconfirmed.

## Q2. Classic findings on CXR shift (confounding, shortcuts, subgroups, label/acquisition shift, corruptions, adversarial)

### Takeaway
Before VLMs, the literature had already established that CXR models (a) learn hospital and device signatures and lose AUC off-site, (b) exploit shortcuts such as text markers, laterality tokens, chest drains and dataset source, (c) underdiagnose under-served subgroups and encode protected attributes, and (d) suffer from label-definition shift between labelers and sites. Corruption and adversarial studies were mostly done on classifiers.

### Cited Findings
- **Confounding / site shift.** Zech et al. 2018 (PLOS Medicine) trained pneumonia CNNs on NIH, Mount Sinai and Indiana data. Performance was significantly lower on external hospitals in 3 of 5 comparisons. A CNN could identify the source hospital system with near-perfect accuracy, and pneumonia prevalence differed by site, so a model could exploit "which hospital" as a proxy. [known] — [DOI 10.1371/journal.pmed.1002683](https://doi.org/10.1371/journal.pmed.1002683)
- **Shortcut learning (COVID).** DeGrave, Janizek & Lee 2021 (Nature Machine Intelligence) found that COVID-19 CXR classifiers trained on composite datasets relied on spurious cues: laterality markers, text, patient positioning and image borders. Performance fell sharply on external data; saliency and GAN-based counterfactuals exposed the shortcuts. [known] — [DOI 10.1038/s42256-021-00338-7](https://doi.org/10.1038/s42256-021-00338-7)
- **Hidden stratification.** Oakden-Rayner et al. 2020 showed that pneumothorax models on NIH14 performed well mainly on cases that already had chest drains, meaning treated cases. [known] — [arXiv 1909.12475](https://arxiv.org/abs/1909.12475)
- **Shortcut exploitation and prevention.** Jabbour et al. 2020 (MLHC) showed that CXR models learn patient attributes (age, sex, BMI) as shortcuts. [known] — [arXiv 2009.10132](https://arxiv.org/abs/2009.10132)
- **Cross-domain generalization and label/concept shift.** Cohen et al. 2020 (MIDL) argued that poor cross-dataset generalization is driven largely by *label* (concept) shift between datasets and labelers, not only image shift. [known] — [arXiv 2002.02497](https://arxiv.org/abs/2002.02497). Pooch et al. 2020 measured AUC drops for classifiers moved between ChestX-ray14, CheXpert and MIMIC-CXR. [known] — [arXiv 1909.01940](https://arxiv.org/abs/1909.01940)
- **Subgroup underdiagnosis.** Seyyed-Kalantari et al. 2021 (Nature Medicine) studied classifiers on MIMIC-CXR, CheXpert and NIH14. Their models underdiagnosed (higher false "no finding" rates for) female patients, patients aged 0–20, Black and Hispanic patients and Medicaid-insured patients, with compounding at intersections. [known] — [DOI 10.1038/s41591-021-01595-0](https://doi.org/10.1038/s41591-021-01595-0)
- **Race recognition.** Gichoya et al. 2022 (Lancet Digital Health) showed that deep models predict self-reported race from CXRs with high AUC, even from degraded or cropped images. [known] — [DOI 10.1016/S2589-7500(22)00063-2](https://doi.org/10.1016/S2589-7500(22)00063-2)
- **Encoding of protected characteristics.** Glocker et al. 2023 (eBioMedicine) found that disease-detection models encode race and sex, but this does not by itself explain subgroup performance gaps. [known] — [DOI 10.1016/j.ebiom.2023.104467](https://doi.org/10.1016/j.ebiom.2023.104467)
- **Fairness under real-world generalization.** Yang et al. 2024 (Nature Medicine), "The limits of fair medical imaging AI in real-world generalization," showed that models using more demographic shortcuts had larger fairness gaps. Fairness "fixed" in-distribution did not transfer to OOD sites. [known] — [DOI 10.1038/s41591-024-03113-4](https://doi.org/10.1038/s41591-024-03113-4)
- **Acquisition shift (AP/PA, portable, ICU).** In Flamingo-CXR, clinician preference for AI reports was much lower for ICU/portable films (56.1%) than for in/outpatient films (77.7%), and higher still for normal cases (94%). This is an acquisition and case-mix effect within the same system. [fetched] — [Tanno et al., Nat Med 2024, DOI 10.1038/s41591-024-03302-1](https://doi.org/10.1038/s41591-024-03302-1)
- **Synthetic corruptions (classifier era).** MedMNIST-C (Di Salvo, Doerrich & Ledig, 2024) applies modality-specific corruptions at 5 severities across 12 MedMNIST+ datasets and 9 modalities, ChestMNIST included. Corruption families include digital (JPEG, pixelate), noise, blur, colour/contrast and task-specific artefacts. The paper also shows that the corruptions work as augmentation that beats generic augmentation for robustness. Code is `medmnistc-api` and data are on Zenodo 11471504. [fetched] — [arXiv 2406.17536](https://arxiv.org/abs/2406.17536)
- **Adversarial perturbations.** Finlayson et al. 2019 (Science) demonstrated adversarial attacks on medical classifiers, CXR included. [known] — [DOI 10.1126/science.aaw4399](https://doi.org/10.1126/science.aaw4399). Ma et al. 2021 (Pattern Recognition) found medical image DNNs *more* vulnerable to adversarial attacks than natural-image models, though the attacks were also easier to detect. [known] — [arXiv 1907.10456](https://arxiv.org/abs/1907.10456)

### Inferences
- These categories (site, subgroup, label definition, acquisition/view, corruption, adversarial) give a ready-made shift taxonomy that can be applied to report-generation VLMs. All of them were established mainly with classifiers.

### Gaps
- Exact AUC drop numbers for Zech 2018 and DeGrave 2021 were not re-fetched, so verify them before quoting.

## Q3. VLM-specific robustness studies on CXR / medical imaging (2022–2026), with numbers

### Takeaway
Robustness evidence for VLMs is fragmented. Zero-shot CLIP-style CXR models (CheXzero) transfer reasonably to PadChest. General and medical VLMs lose about 10% accuracy under even weak artefacts and almost never flag strong artefacts. Report generators show lexical-metric drops of 3–15 points on OOD sites, and at least one documented case (MAIRA-2 on Mayo data) of poor institutional generalization. No study systematically measures *grounding/faithfulness* of report generators under controlled shift.

### Cited Findings
**Zero-shot CLIP-style models**
- CheXzero (Tiu et al., Nature BME, 15 Sep 2022) was zero-shot on PadChest: 39,053 radiographs, 57 findings with >50 cases. AUC was ≥0.900 for 6 findings and ≥0.700 for 38 more. The model was trained only on MIMIC-CXR image–report pairs. [fetched via press coverage] — [AuntMinnie](https://www.auntminnie.com/clinical-news/article/15631856/new-ai-model-doesnt-require-annotated-training-data); paper DOI [10.1038/s41551-022-00936-9](https://doi.org/10.1038/s41551-022-00936-9) [known]
- A 2026 Nature BME supplement (the CLEAR model paper) re-reports CheXzero's zero-shot PadChest AUROCs: pleural effusion 0.965, pulmonary mass 0.883, consolidation 0.875, heart insufficiency 0.924, hilar congestion 0.888. CLEAR is higher on most of these. [fetched] — [Springer ESM s41551-026-01741-4](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41551-026-01741-4/MediaObjects/41551_2026_1741_MOESM1_ESM.pdf)
- Demographic bias of CheXzero-style foundation models (underdiagnosis of female, Black and intersectional groups) is reported in "Demographic bias of expert-level vision-language foundation models in medical imaging" (Yang et al., Science Advances 2025). The DOI is **UNVERIFIED**.

**Corruption / artefact robustness of VLMs**
- Cheng, Zhou et al. 2025 (npj Digital Medicine, published 27 Nov 2025) evaluated GPT-4o, Claude 3.5 Sonnet (two versions), Llama 3.2 11B, BiomedCLIP and MedGemma on brain MRI, CXR and retina. The CXR data were a Kaggle COVID-19 detection set, 100 diseased and 100 normal images. Artefacts were bias field, noise and motion (intensity) and crop and rotation (spatial), each at weak and strong scales. Results on CXR [fetched] — [PMC12660864](https://pmc.ncbi.nlm.nih.gov/articles/PMC12660864/); [DOI 10.1038/s41746-025-02108-w](https://doi.org/10.1038/s41746-025-02108-w):
  - Mean clean accuracy was 0.604. BiomedCLIP was best at 0.760 (95% CI 0.710–0.805).
  - Weak artefacts cut accuracy by 10.46% on average. BiomedCLIP under noise fell to 0.466, a 38.5% drop (p<0.001).
  - The mean detection rate for strong artefacts was only 0.115, so models rarely noticed corrupted inputs.
  - GPT-4o sensitivity under motion artefacts fell from 0.809 (structured output) to 0.331 (CoT), partly because of refusals.
  - Claude models gained sensitivity under artefacts, which the authors read as artefacts being mistaken for lesions.
  - MedGemma had the lowest CXR accuracy. Its reasoning-vs-answer conflict rate rose from 0.204 on clean images to 0.427 under weak motion.
- MediMeta-C / RobustMedCLIP (Imam, Marew & Yaqub; MIUA 2025) is a new corruption benchmark used alongside MedMNIST-C, covering 5 modalities and 5 medical VLMs. Existing models "exhibit severe degradation under corruption." Few-shot LoRA adaptation of the visual encoder improves robustness. [fetched] — [arXiv 2505.15425](https://arxiv.org/abs/2505.15425)
- MedQ-Deg (2026) is a multidimensional benchmark of MLLMs across medical image-quality degradations. CXR-specific numbers were not confirmed. [fetched: existence] — [arXiv 2603.07769](https://arxiv.org/abs/2603.07769)
- VLM-RobustBench (2026) is a general, non-medical VLM corruption benchmark that adds resampling and watermark families. It is useful as a protocol template. [fetched: existence] — [arXiv 2603.06148](https://arxiv.org/abs/2603.06148)

**Report generation under shift**
- MAIRA-2 (Microsoft; grounded report generation) was evaluated on PadChest as an external set, giving logical precision/recall of 57%/49%. Lexical scores were much lower (ROUGE 28%, BLEU-4 10%), which the authors attribute partly to PadChest lacking indication, technique and comparison sections. [fetched via search snippet] — [arXiv 2406.04449](https://arxiv.org/abs/2406.04449)
- A 2025 structured-report-generation paper tested CheXpert-Plus models, CheXagent and MAIRA-2 on an OOD HOPPR test set. BLEU, ROUGE-L and BERTScore fell by about 3–15 points against in-distribution results. Structure-aware metrics were more stable, though RadGraph-F1 also declined. [fetched via search snippet] — [arXiv 2505.24223](https://arxiv.org/abs/2505.24223)
- A longitudinal-report evaluation reported that MAIRA-2 "does not generalize well to the Mayo Clinic institutional dataset." [fetched via search snippet; attribution to the exact paper is uncertain] — [arXiv 2601.16753](https://arxiv.org/abs/2601.16753)
- ReXrank (Zhang et al., Nov 2024) is a public leaderboard that scores report generators on ReXGradient (10k private studies), MIMIC-CXR, IU-Xray and CheXpert Plus with 8 metrics, separating findings-only from findings+impression models. It is effectively a multi-site comparison, but it reports per-dataset scores, not a controlled shift analysis. [fetched] — [arXiv 2411.15122](https://arxiv.org/abs/2411.15122)
- Flamingo-CXR (Nature Medicine 2024) gave an expert evaluation on IND1 (India) and MIMIC-CXR. Clinically significant errors were found in AI-only reports for 22.8% of in/outpatient cases, human-only reports for 14.0%, and both for 24.8%. [fetched] — [DOI 10.1038/s41591-024-03302-1](https://doi.org/10.1038/s41591-024-03302-1); [arXiv 2311.18260](https://arxiv.org/abs/2311.18260)
- "Radiology Report Generation for Low-Quality X-Ray Images" (Zhu et al., Apr 2026) uses a quality-assessment agent to pull low-quality MIMIC-CXR studies into an LRRG benchmark. It states that existing VLMs show "severe performance degradation" on suboptimal images and proposes dual-loop bi-level training. The abstract gives no numbers and no faithfulness or grounding metrics. [fetched] — [arXiv 2604.10188](https://arxiv.org/abs/2604.10188)
- "Reporting Practice Matters" (2026) shows that the choice of reference report changes CXR report-evaluation scores. This is relevant because reporting-style shift across sites confounds metric drops. [fetched: existence] — [arXiv 2609.19093](https://arxiv.org/abs/2609.19093)
- CheXagent (arXiv 2401.12208) reports CheXbench (8 tasks) and a sex/race/age fairness analysis, but no dedicated OOD-site study. LLaVA-Med (arXiv 2306.00890) reports only in-house GPT-4-scored chat benchmarks (CXR column 56.2 vs 41.6 for LLaVA). [fetched] — [CheXagent](https://arxiv.org/abs/2401.12208); [LLaVA-Med](https://arxiv.org/abs/2306.00890)

**Metric robustness (relevant for faithfulness measurement under shift)**
- GREEN (EMNLP Findings 2024) correlates with ReXVal expert error counts better than RadGraph (0.63 vs 0.47). It degrades zero-shot on non-CXR data, and others report a length bias. [fetched] — [arXiv 2405.03595](https://arxiv.org/abs/2405.03595)
- Med-Gemini notes that RadGraph-based metrics are tied to MIMIC-style reports and were not computed on IND1. This is a metric-side shift problem. [fetched] — [arXiv 2405.03162](https://arxiv.org/abs/2405.03162)

### Inferences
- Lexical metrics confound true degradation with reporting-style shift (the PadChest and HOPPR results above). A shift study for report generation should therefore lean on clinical, label-level, grounding-level and expert-aligned metrics, and control for reference style.
- The ~0.115 artefact-detection rate suggests VLMs give confident outputs on corrupted CXRs. A study of *faithfulness under corruption* (does the model still describe findings it can no longer see?) would be new.

### Gaps
- I found no "RadRobust"-named work. Any such title is **UNVERIFIED** and probably does not exist.
- No study gave CheXagent or LLaVA-Med report-generation numbers on BRAX, VinDr-CXR or CANDID-PTX.
- Full numeric tables for MedQ-Deg, MediMeta-C and arXiv 2505.24223 were not fetched.

## Q4. Has "distribution shift for report-generation VLMs, measured with grounding/faithfulness" been done? Closest prior work and what it lacks

### Takeaway
I found no paper that systematically applies controlled shifts (site, acquisition, corruption, demographic) to CXR *report-generation* VLMs and measures *grounding* (box/phrase alignment) and *faithfulness* (hallucination or unsupported findings) as the outcome. The closest works each cover only one or two of these three pieces: shift, report generation, and grounding/faithfulness.

### Cited Findings
- **LRRG (arXiv 2604.10188):** report generation under image-quality shift. It covers MIMIC-CXR only, lists no faithfulness or grounding metrics in the abstract, and studies no site shift. [fetched] — [arXiv 2604.10188](https://arxiv.org/abs/2604.10188)
- **MAIRA-2 (arXiv 2406.04449):** grounded report generation with a PadChest external test. Grounding is evaluated in-distribution, and the external test reports only lexical and logical metrics, with no analysis by shift type. [fetched] — [arXiv 2406.04449](https://arxiv.org/abs/2406.04449)
- **Structured RRG with HOPPR OOD (arXiv 2505.24223):** one OOD set, with lexical, structure and RadGraph metrics. It has no grounding and no controlled shift taxonomy. [fetched] — [arXiv 2505.24223](https://arxiv.org/abs/2505.24223)
- **ReXrank (arXiv 2411.15122):** multi-dataset leaderboard with no shift attribution and no grounding. [fetched] — [arXiv 2411.15122](https://arxiv.org/abs/2411.15122)
- **Cheng et al. npj DM 2025 and MediMeta-C:** artefact and corruption robustness of VLMs, but on classification/VQA accuracy, not report generation or faithfulness. [fetched] — [PMC12660864](https://pmc.ncbi.nlm.nih.gov/articles/PMC12660864/); [arXiv 2505.15425](https://arxiv.org/abs/2505.15425)
- **Grounding and hallucination tools:** these are in-distribution only, but usable as faithfulness metrics under shift. Examples are FactCheXcker (measurement hallucinations, [arXiv 2411.18672](https://arxiv.org/abs/2411.18672)), a phrase-grounded fact-checker ([MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/3526_paper.pdf)), RadAlign (external OpenI classification AUC 0.923 vs 0.836, [MICCAI 2025](https://papers.miccai.org/miccai-2025/0741-Paper3308.html)) and GREEN ([arXiv 2405.03595](https://arxiv.org/abs/2405.03595)). [fetched]
- **PadChest-GR** supplies bilingual grounded reports that could serve as an external *grounding* benchmark. [fetched: existence] — [arXiv 2411.05085](https://arxiv.org/abs/2411.05085)

### Inferences
- The open niche is a controlled benchmark that combines four elements: (i) MIMIC-trained or public report-generation VLMs (CheXagent, MAIRA-2, LLaVA-Rad/LLaVA-Med, MedGemma, etc.), (ii) natural shifts (CheXpert Plus, IU-Xray, PadChest-GR, ReXGradient) plus synthetic shifts (MedMNIST-C-style corruptions, AP/PA and portable subsets, demographic subgroups), (iii) grounding metrics against boxes (PadChest-GR, VinDr-CXR, MS-CXR), and (iv) faithfulness and hallucination metrics (GREEN, RadGraph, FactCheXcker-style checks, "does the model notice the artefact"). Each piece exists separately; I did not find them combined.
- A risk to flag: metric shift. Report-quality metrics trained on MIMIC (RadGraph, CheXbert) may themselves degrade on PadChest or translated text, so part of any measured drop could be an artefact of the metric.

### Gaps
- My search coverage of 2026 preprints (Jan–Oct 2026) is limited to what turned up in about 15 queries. A closer competitor may exist and should be checked with targeted arXiv searches ("report generation" + "distribution shift" / "domain shift" / "corruption" + "grounding").
- MS-CXR (phrase grounding, Microsoft) was not re-verified this session.
