# 02 — Related Work and Positioning

## 1. MedThink (the work you build on)

Gai et al., *MedThink: A Rationale-Guided Framework for Explaining Medical VQA*, Findings of NAACL 2025 (local copy: `2025.findings-naacl.415.pdf`).

- **Data:** R-RAD, R-SLAKE and R-Path are VQA-RAD, SLAKE and PathVQA with GPT-4V-generated MDMRs (rationales), checked by physicians on coherence, relevance and accuracy.
- **Model:** UnifiedQA T5-base (223M) + DETR visual features, single-head cross-attention, sigmoid gate fusion (Eq. 2–4). Trained with standard token-level NLL.
- **Strategies:** w/o-R, Explanation (A→R), Reasoning (R→A), Two-Stage (R model → A model).
- **Claims:** Explanation gives +4.5 / +3.8 / +1.0 accuracy on RAD / SLAKE / Path; Reasoning hurts; 0.2B params is competitive with 7B MLLMs.
- **Limitations relevant to you:**
  - English only.
  - Generic COCO DETR features.
  - Single seed and no variance reported.
  - The open-end evaluation is n-gram overlap on answer+rationale.
  - No analysis of why Reasoning hurts. Your loss-dilution analysis covers that gap.

## 2. Bangla Med-VQA: direct competitors (read these before writing)

| Work | What it is | How you differ |
|---|---|---|
| *Evaluating Large Vision Language Models on Bangla Medical VQA* (Findings of ACL 2026) — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1862/) | Clinically validated Bangla Med-VQA dataset; benchmarks 9 LVLMs zero-shot, with CoT, and with LoRA; Bangla is weaker than English | **No rationale annotation, no parallel controlled study, focus on large LVLMs.** You offer rationale supervision for a 0.25B model, plus a mechanism analysis. |
| *How Good LLMs Are at Answering Bangla Medical Visual Questions? Dataset and Benchmarking* (arXiv 2605.18111) — [arXiv](https://arxiv.org/html/2605.18111) | Bangla Med-VQA built from MedICaT + ChestX-ray14; LVLM benchmarking | Different sources (yours are the standard VQA-RAD/SLAKE/PathVQA), **no rationales**, benchmark-only |
| Med-VQA_Bn_Overall (≈7k items, mentioned in the above) | Small Bangla medical VQA set | Smaller, no rationales |

General (non-medical) Bangla VQA: BengaliVQA ([Mendeley](https://data.mendeley.com/datasets/y9fw6k37n9/1)), Bangla-Bayanno ([arXiv 2508.19887](https://arxiv.org/pdf/2508.19887)), ChitroJera ([ResearchGate](https://www.researchgate.net/publication/396003588_ChitroJera_A_Regionally_Relevant_Visual_Question_Answering_Dataset_for_Bangla)).

**Action:** Download and read both Bangla Med-VQA papers this week. If either has released data, add a zero-shot or transfer evaluation of your best model on it. That would be a strong generalisation result.

## 3. Multilingual / rationale-based medical reasoning (cite for framing)

- **S-Chain** (arXiv 2510.22728): 12k expert-annotated images with stepwise visual CoT, extended to 700k VQA pairs in 16 languages. It shows the value of *expert-verified* multilingual rationales, which supports your human-checked data. [link](https://www.arxiv.org/pdf/2510.22728)
- **Med-CoReasoner / MultiMed-X** (arXiv 2601.08267): documents an English-vs-local-language reasoning gap that is larger for low-resource languages, and notes that MT of reasoning is unreliable. This directly supports your F1 finding. [link](https://arxiv.org/html/2601.08267)
- **Multilingual Medical Reasoning for QA** (arXiv 2512.05658): multilingual reasoning traces for text QA. [link](https://arxiv.org/pdf/2512.05658)
- **ChexReason** (arXiv 2512.23090): small VLM, SFT + GRPO with reasoning traces on chest X-ray. Represents the "small model + reasoning" line. [link](https://arxiv.org/pdf/2512.23090v1)
- MM-CoT (Zhang et al., 2023): the architecture MedThink inherits. Its finding that rationale-first generation hallucinates without vision is the English analogue of your F3.
- The "distilling step-by-step" / rationale-as-multi-task literature (Hsieh et al., 2023): rationales as an auxiliary task with **separately weighted** losses. This directly motivates your loss fix.
- Medical visual encoders: BiomedCLIP (Zhang et al., 2023), PubMedCLIP (Eslami et al., 2023), MedSigLIP (Google, 2025).

> Literature check performed 2026-11-20 with web search. The competitor list may not be exhaustive. Run a Google Scholar search for "Bangla/Bengali medical VQA" and "multilingual medical VQA rationale" before submission.

## 4. The gap you fill (positioning statement)

> Existing Bangla Med-VQA work benchmarks large LVLMs on answer accuracy. No prior work provides **rationale-annotated** Bangla Med-VQA data, and no prior work asks whether **rationale supervision**, which helps small English models, **transfers to a low-resource language**. We show that it does not, explain why (answer-loss dilution, weak visual grounding, low-resource decoder capacity), and propose a lightweight fix that recovers accuracy while keeping clinician-rated explanations. The model runs at 0.25B parameters on a single free-tier GPU.

The "free-tier GPU / 0.25B" framing is an asset for a Q1 medical-informatics journal (CIBM, AIIM, JBI). Deployability in low-resource health systems is an impact argument reviewers there value.
