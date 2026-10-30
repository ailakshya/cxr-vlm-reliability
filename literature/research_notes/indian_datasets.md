# Chest X-ray datasets with Indian-hospital / Indian-population data (2015–2026)

Research date 2026-10-29. "VERIFIED" means the dataset/landing page was fetched in this session. "UNVERIFIED" means it came only from a search snippet or from prior knowledge.

## Q1. Public or request-access CXR datasets from India

### Takeaway
Very little Indian CXR data is public. Three releases can actually be downloaded or requested: **IN-CXR** (ICMR-NIRT, national TB prevalence survey, normal/abnormal labels only, registration required), the **Mendeley "COVID Chest X-Rays"** set (799 images from KGMU Lucknow, UPUMS Saifai, and GMC Kota; CC BY 4.0), and the small 2014 **NITRD New Delhi TB "Dataset A/B"** (306 images, SourceForge). **TB Portals** (NIAID) also holds some Indian TB CXRs behind a data-access process. The largest Indian collections are private. These include IND1 from Apollo Hospitals (used by Google), Qure.ai's sets, and the 5M-image multi-site set from arXiv 2504.00022.

### Cited Findings

**A. Public / request-access (VERIFIED)**

- **IN-CXR (ICMR–National Institute for Research in Tuberculosis, Chennai).** The images are adult PA chest radiographs taken across India in the National TB Prevalence Survey (2019–21). They are grouped only as *normal* vs *abnormal* and come as DICOM. Access is by registering on the IN-CXR portal (intbcxr.nirt.res.in). The page gives no image count, licence, DUA text, or DOI/paper. It mentions no reports, bounding boxes, or finding-level labels. — [NIRT IN-CXR page](https://nirt.res.in/incxr)
  - An older NIRT URL (nirt.res.in/html/xray.html) now returns 404. — [NIRT xray page (404)](https://www.nirt.res.in/html/xray.html)
- **COVID Chest X-Rays (Mendeley Data, DOI 10.17632/4n66brtp4j.1, published 19 Jan 2021, CC BY 4.0).** There are 799 images in total: KGMU Lucknow 188 (28 COVID, 83 abnormal, 77 healthy), UPUMS Saifai/Etawah 68 (all COVID), and Government Medical College Kota, Rajasthan 543 (326 COVID, 217 abnormal). That makes 422 COVID, 300 abnormal, and 77 healthy. Labels are image-level classes only. The contributor is Malay Kishore Dutta, and the associated paper is Joshi RC et al., 2021, *Biocybernetics and Biomedical Engineering*. Open download. — [Mendeley 4n66brtp4j](https://data.mendeley.com/datasets/4n66brtp4j/1)
- **NITRD New Delhi TB datasets "DA" and "DB" (Chauhan, Chauhan & Rout, PLoS ONE 2014, DOI 10.1371/journal.pone.0112980).** These come from the National Institute of TB and Respiratory Diseases, New Delhi. DA has 156 images (78 TB / 78 non-TB), taken on a Diagnox-4050 and digitised with AGFA CR35-X, 8-bit. DB has 150 images (75/75), taken on a PRORAD URS with Canon CXDI, 14-bit. Labels are image-level TB vs non-TB only. The paper says they are freely available on SourceForge. — [PLoS ONE paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0112980)
  - The SourceForge folder exists, with "Dataset A" and "Dataset B" folders (2014) and a label_information.txt file. No licence is stated. — [SourceForge tbxpredict/data](https://sourceforge.net/projects/tbxpredict/files/data/)
  - Note: published in 2014, so just outside the 2015–2026 window, but still widely used (e.g. in arXiv 2011.09778, "Deep Learning for Automated Screening of TB from Indian Chest X-rays") — [arXiv 2011.09778](https://arxiv.org/abs/2011.09778); also called "India TB CXR" in [arXiv 2102.10607](https://arxiv.org/pdf/2102.10607)
- **TB Portals (NIAID).** Radiological data come from institutions in India, Belarus, Romania, Georgia, and Azerbaijan, with Belarus the largest. The only Indian contributor named is the College of Pharmaceutical Sciences, Manipal (mainly drug-sensitive TB). The data include radiology reports, radiologist image annotations, and partial AI annotations. Data are "made publicly available after quality checks", and access goes through the portal's Access Data process (in practice a data-use agreement; the exact terms were not read). No India-specific image count is published. — [TB Portals sources](https://tbportals.niaid.nih.gov/where-do-our-cases-come-from)
  - Scale: more than 4,000 cases and 4,000 CXRs as of April 2021. A later study reported 10,953 CXRs from 8,848 participants (snippet only). — [search snippet via TB Portals / medRxiv 2022.04.21.22273975](https://www.medrxiv.org/content/10.1101/2022.04.21.22273975v1.full)

**B. Private Indian data used in papers (VERIFIED as described in the papers)**

- **IND1 (Google; Flamingo-CXR, Med-Gemini, and earlier Google CXR work).** 263,021 frontal CXRs (digital and scanned) with free-text reports. Flamingo-CXR describes it as a "private research dataset" from five regional centres of a large Indian hospital group (Bangalore, Bhubaneswar, Chennai, Hyderabad, New Delhi), citing Nabulsi et al. 2021. Splits were 250,066 train, 4,960 val, and 7,995 test. A 2,306-case subset has binary labels for six conditions from US radiologists. Collection ran Nov 2010–Jan 2018. — [Flamingo-CXR, arXiv 2311.18260](https://arxiv.org/pdf/2311.18260); also used in [Med-Gemini, arXiv 2405.03162](https://arxiv.org/pdf/2405.03162)
  - The hospital group is Apollo Hospitals. Google's blog says the model was trained on more than 200,000 de-identified CXRs "from the Apollo Hospitals in India", and Nabulsi et al. (Sci Rep 2021, DOI 10.1038/s41598-021-93967-2) has a co-author from Apollo Radiology International, Hyderabad. The test split is called DS-1. — [Google Research blog](https://www.research.google/blog/detecting-abnormal-chest-x-rays-using-deep-learning/); [Nabulsi et al. PMC8410908](https://pmc.ncbi.nlm.nih.gov/articles/PMC8410908)
  - Access: private, not available.
- **Qure.ai CQ2000 / CQ100k (Putha et al., arXiv 1807.07455).** CQ2000 has 2,000 CXRs from three Columbia Asia Hospitals (Kolkata, Pune, Mysore), with a 3-radiologist majority reference. CQ100k and the development set are Indian data. Nothing says the data were released. — [arXiv 1807.07455 (ar5iv)](https://ar5iv.arxiv.org/html/1807.07455). Private.
- **Multi-site Indian autonomous-AI study (arXiv 2504.00022).** Trained on more than 5 million CXRs from "healthcare systems across India". Deployed in 17 Indian healthcare systems with more than 150,000 scans processed. The abstract does not name the company or sites (likely 5C Network, but UNVERIFIED). No public-data statement. — [arXiv 2504.00022](https://arxiv.org/abs/2504.00022)

**C. UNVERIFIED leads (search-snippet only)**

- **Deep LF-Net (arXiv 2011.09695).** Lung segmentation on "Indian chest radiographs including severely unhealthy images". If the masks were released, this would be the only Indian segmentation set found. Availability not checked. — [arXiv 2011.09695](https://arxiv.org/pdf/2011.09695)
- **CMC Vellore in-house set.** 688 CXRs (216 healthy, 472 clinically confirmed patients) appear in one study; no release found. UNVERIFIED (snippet only, source paper not identified).
- **MIDRC.** A search snippet said it includes COVID CXRs "from South Korea, India, and the USA". I did not confirm this on MIDRC pages. UNVERIFIED. — [snippet from MDPI Diagnostics 16(1):146](https://www.mdpi.com/2075-4418/16/1/146)
- **Kaggle "Chest X-Rays Tuberculosis from India".** Cited in a 2025 paper, but the Kaggle page was not found. It is probably a re-upload of NITRD DA/DB. UNVERIFIED.
- **AIKosh / IndiaAI dataset platform.** Searches returned no CXR dataset. No evidence either way; checking the portal directly is required. UNVERIFIED.
- **PGIMER, AIIMS, KGMU (beyond the Mendeley COVID set).** Searches found no public CXR releases.

**D. Look-alikes that are NOT Indian (exclusions)**

- Mendeley TB CXR set 8j2g3csprk (2,494 TB + 514 normal, CC BY 4.0) is from Pakistan, as are Mendeley 8cm4ffk7b4 (pneumonia) and y6dvssx73b (Multan). — [Mendeley 8j2g3csprk](https://data.mendeley.com/datasets/8j2g3csprk)
- The Shenzhen/Montgomery sets are China/USA, not India. — [Jaeger et al. QIMS 2014](https://qims.amegroups.org/article/view/5132)
- Multinational CXR benchmark arXiv 2505.16027: 12 datasets from six countries, no Indian dataset named in the abstract. — [arXiv 2505.16027](https://arxiv.org/abs/2505.16027)

### Inferences
- From prior knowledge, not re-verified here: RSNA Pneumonia is NIH ChestX-ray14 (US), SIIM-FISABIO-RSNA COVID is BIMCV (Spain), and VinDr is Vietnam. None of them has Indian hospital sites. COVIDx and Cohen's covid-chestxray-dataset collect images from publications and Radiopaedia, and some of those may be Indian cases. The country of origin is not systematically tagged, so these are not usable as an "Indian site".
- IN-CXR is the only government-backed, population-scale Indian release. Because it is a TB *screening survey* (community, PA view), its distribution differs from hospital CXRs, which makes it a useful shift test site in itself.

### Gaps
- The IN-CXR image count, licence, DUA terms, and whether any finding-level labels exist beyond normal/abnormal are not stated publicly. Registering is the only way to find out.
- Number of Indian-origin CXRs in TB Portals is unknown.
- AIKosh catalogue not checked directly (no search hits).

## Q2. Does any Indian CXR dataset include radiology reports or bounding boxes?

### Takeaway
I found no *public* Indian CXR dataset with free-text reports or bounding boxes. Reports exist only in private sets (IND1/Apollo with 263k reports, the Qure.ai and 5C-style corporate data). TB Portals has reports and radiologist annotations, but it is mostly Belarus, with only a small, uncounted Indian share.

### Cited Findings
- IND1 has paired reports but is a "private research dataset". — [arXiv 2311.18260](https://arxiv.org/pdf/2311.18260)
- IN-CXR: only normal/abnormal grouping is described. — [NIRT IN-CXR](https://nirt.res.in/incxr)
- Mendeley COVID set and NITRD DA/DB: image-level class labels only. — [Mendeley](https://data.mendeley.com/datasets/4n66brtp4j/1); [PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0112980)
- TB Portals: reports plus radiologist annotations, multi-country including India. — [TB Portals](https://tbportals.niaid.nih.gov/where-do-our-cases-come-from)
- A search for 2024–25 Indian CXR sets with reports or boxes returned only PadChest-GR (Spain) and VinDr-CXR (Vietnam). — [PadChest-GR arXiv 2411.05085](https://arxiv.org/pdf/2411.05085)

### Inferences
- For a VLM report-generation or grounding evaluation on Indian data, the realistic options are classification-style probes (normal/abnormal, TB yes/no, COVID) phrased as VQA or zero-shot prompts. A report-level evaluation would need an Indian hospital collaboration.

### Gaps
- Whether Deep LF-Net (arXiv 2011.09695) released its Indian lung masks was not checked.

## Q3. Best realistic option for an Indian student as the "Indian hospital" shift site

### Takeaway
The first choice is **IN-CXR** (register at NIRT; national, multi-region, DICOM, normal/abnormal). The best immediately available fallback is the **Mendeley KGMU/UPUMS/GMC-Kota COVID set** (799 images, CC BY 4.0, hospital data, 3 Indian sites). **NITRD DA/DB** (306 TB images) can serve as a tiny second Indian TB site. Together they give two or three Indian sources with image-level labels, but no reports or boxes.

### Cited Findings
- IN-CXR: registration-based access, national 2019–21 survey, DICOM. — [NIRT IN-CXR](https://nirt.res.in/incxr)
- Mendeley COVID set: CC BY 4.0, open, 799 images from 3 Indian hospitals. — [Mendeley](https://data.mendeley.com/datasets/4n66brtp4j/1)
- NITRD DA/DB: free on SourceForge, 2 X-ray machines, which is useful for a scanner-shift probe. — [PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0112980); [SourceForge](https://sourceforge.net/projects/tbxpredict/files/data/)

### Inferences
- The Mendeley set is CC BY 4.0, so it can be cited and redistributed without a DUA, which makes it the safest choice for a paper deadline. Its weakness is the small size and the COVID-era label set (COVID / abnormal / healthy). The "abnormal" and "healthy" classes support a normal-vs-abnormal probe that lines up with IN-CXR.
- IN-CXR is the strongest scientifically, but access time and licence are unknown. Register early, and state the DUA terms in the paper.
- Private sets (IND1, Qure.ai, 5C) are useful only as related-work citations showing that big labs already test on Indian data.

### Gaps
- IN-CXR approval turnaround and redistribution terms are unknown.
