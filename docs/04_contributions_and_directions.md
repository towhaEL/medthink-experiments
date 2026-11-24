# 04 — Proposed Contributions and Research Directions

## Working title

**"When Rationales Hurt: Diagnosing and Repairing Rationale-Supervised Medical VQA for Low-Resource Bangla"**

Alternative: *"BnMedThink: A Human-Verified Rationale Benchmark and Answer-Anchored Learning for Bangla Medical VQA"*

## The paper's story in four sentences

1. Rationale supervision (MedThink) improves small English Med-VQA models. Low-resource-language clinical AI needs both small models and explanations.
2. We release the first **human-verified, rationale-annotated, EN–BN parallel** Med-VQA benchmark (≈39k QA + rationales over 3 datasets).
3. Under a controlled same-pipeline comparison, rationale supervision **reverses** in Bangla (it hurts on all datasets). We trace this to (i) answer-loss dilution under limited decoder capacity, (ii) near-closed visual fusion, and (iii) generic visual features.
4. A lightweight fix, **Answer-Anchored Rationale Learning (AARL)**, combines segment-balanced loss, a medical visual encoder and gate warm-up. It recovers or exceeds w/o-R accuracy while producing rationales that clinicians rate higher, at 0.25B params on a free-tier GPU.

## Contributions (as they would appear in the paper)

| # | Contribution | Type | Evidence needed | Status |
|---|---|---|---|---|
| C1 | **BnMedThink benchmark**: R-RAD-Bn, R-SLAKE-Bn, R-Path-Bn, human-verified translation of questions, answers and rationales; parallel to English; fixed train/val/test splits | Resource | Translation QA statistics + 300-item double audit (doc 06) | Data done, stats missing |
| C2 | **Controlled cross-lingual study** showing the rationale benefit does not transfer | Empirical finding | E5 (same-pipeline English control), 3 seeds, significance tests | Partly (BN side done) |
| C3 | **Mechanistic diagnosis**: answer-loss dilution, visual under-use, rationale-answer leakage | Analysis | E0 (blind), gate stats, E4 (rationale length), loss curves | Partly (evidence in doc 01) |
| C4 | **AARL**: segment-balanced loss + BiomedCLIP features + gate warm-up | Method | E2, E3 ablations; final model on all 3 datasets × 2 subsets | To do |
| C5 | **Evaluation suite**: Bengali-aware metrics, choice scoring, seen/unseen answer split, rationale–answer consistency, clinician ratings, zero-shot VLM baselines | Evaluation | E6, E7, human eval | To do |

C1 + C2 + C3 alone would make a solid Q2 paper. **C4 and the human part of C5 are what lift it to Q1.**

---

## Method details: AARL

### (a) Segment-balanced objective
`L = α · L_ans + (1 − α) · L_rat`, each segment **length-normalised per example** (code in doc 03).
- Hypothesis: with α ≈ 0.8–0.9, answer accuracy recovers to w/o-R or above, and the rationale still acts as an auxiliary task. This is the regularisation effect that helped English.
- Sweep α ∈ {0.5, 0.8, 0.95}. α = token-proportional is the current baseline (≈ 0.02).
- Optional, no extra cost: **rationale dropout**. Train each batch with the rationale target on a random 50% of items and answer-only on the rest. This is a strong, simple baseline that reviewers will think of.

### (b) Medical visual grounding
- Replace DETR (COCO) with **BiomedCLIP ViT-B/16** patch tokens (197 × 768), extracted once and cached like the DETR features.
- Fallback if BiomedCLIP is a problem on Kaggle: PubMedCLIP ViT-B/32 (50 × 768).

### (c) Gate warm-up
- Epoch 0: freeze T5 and train only `image_dense`, `mha_layer` and `gate_dense` (cheap, stable), so the visual pathway learns something before competing with the pretrained text path.
- Ablate gate bias −3 vs −1.
- Report the λ distribution before and after. Opening the gate plus improved blind-ablation gaps is the proof that the model actually uses vision.

### (d) (Optional) Concise rationales
- Train on the first 1–2 sentences of each rationale (E4). If shorter rationales help, that supports the capacity explanation and gives a practical recommendation for low-resource languages.

---

## Research directions considered (and why I rank them as I do)

| Direction | Feasible in 1 month on Kaggle? | Novelty | Verdict |
|---|---|---|---|
| **A. Diagnose + fix rationale reversal (AARL)** | Yes (≈80 GPU-h) | High: nobody has shown this for low-resource Med-VQA | **Main line** |
| B. Cross-lingual rationale transfer: train with English rationales and Bangla answers, or code-switched rationales | Partly. BanglaT5's vocabulary covers English poorly, so this needs mT5-base (2× slower). | High | **Tier 2**: one RAD/SLAKE experiment with mT5 if budget allows. Strong "future work" otherwise. |
| C. LoRA fine-tune a small VLM (Qwen2-VL-2B / PaliGemma-3B) on Bangla rationales | Marginal: 12h sessions and 16GB T4, needs 4-bit QLoRA; R-Path too large | Medium (competitors already LoRA-tune LVLMs) | **Stretch**: QLoRA on RAD only, as a "what a 2B VLM gets" reference row |
| D. LLM-as-judge for Bangla rationales | Needs API (none) or a 7B open model (slow on T4) | Medium | Replace with clinician evaluation (you have 2+ experts, which is better evidence) |
| E. RL (GRPO) for answer-consistent rationales | No (compute) | High | Future work |
| F. Retrieval-augmented rationale generation | Possible but scope creep | Medium | Future work |

## Should we add RAG? (decision, 2026-11-20)

**Not as the main contribution.** RAG adds input context, while the diagnosed causes are in the loss (answer dilution) and the visual pathway (closed gate, COCO features). Adding it alongside AARL makes it impossible to attribute gains. Other problems:
- **Leakage:** 97% of R-RAD test images are also in train. Naive retrieval would return the same image's other QA pairs and rationales.
- **No Bangla medical knowledge corpus** for knowledge-base RAG.
- Longer inputs make training slower on a T4.
- Medical VLM RAG already exists (e.g., MMed-RAG, RULE), so the novelty is lower than the reversal finding.

**Backup plan at Gate 1 (Dec 7):** if the segment-balanced loss does not recover accuracy, the likely cause is that the decoder cannot *generate* Bangla rationales from scratch. Then add **retrieved-case rationale augmentation**:
- Retrieve the top-1/2 training cases by BiomedCLIP image embedding + question similarity, **excluding the same image ID**.
- Append their Bangla rationale to the input as a template (`সদৃশ কেস: …`).
- Reuses the E3a features. Cost: ~2 days of code and ~10 GPU-h (RAD + SLAKE closed, 1 seed, then 3 seeds if it helps).
- Report retrieval with and without same-image exclusion, to show transparently that there is no leakage.

| Idea | Verdict |
|---|---|
| Segment-balanced loss, BiomedCLIP, gate warm-up | Main (AARL) |
| Rationale dropout 50% | Required baseline |
| Rationale length ablation | Analysis |
| Retrieved-case rationale (RAG) | Backup at Gate 1 |
| EN rationale + BN answer (mT5) | Tier 2 |
| QLoRA Qwen2-VL-2B | Optional reference row |
| Knowledge-base RAG, GRPO/RL | Future work |

## Fallback if AARL does not beat w/o-R

The decision gate is at the end of week 2 (see doc 07). If the best AARL variant only **matches** w/o-R:
- Reframe C4 as **"explanations at no accuracy cost"**. The model then gives clinician-rated rationales with no accuracy loss, versus −3 to −15 points for vanilla MedThink. That is still a clear positive result.
- Put more weight on C2/C3 (controlled reversal + mechanism) and C5 (human evaluation).

If AARL does not even close the gap, the paper becomes a resource + analysis paper. In that case, retarget to *Language Resources and Evaluation* (Q1, Springer) or *ACM TALLIP* rather than a medical-AI Q1 journal.
