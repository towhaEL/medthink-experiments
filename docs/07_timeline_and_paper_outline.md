# 07 — Timeline, Decision Gates, Target Journals, Paper Outline

## Timeline (today 2026-11-20 → submission mid-January 2027)

| Week | Dates | Compute (GPU) | Non-compute (you + co-authors) |
|---|---|---|---|
| **W0** | Nov 20–23 | Code changes (doc 03): val split, early stopping, seeds, segment loss, choice scoring, image-ablation flag. Smoke-test with `quick_run`. Start **E3a** (BiomedCLIP features). | Read both Bangla Med-VQA competitor papers. Request MedGemma access on HF. Find out whether the raw LLM translations were kept. |
| **W1** | Nov 24–30 | **E1** baselines (3 seeds), **E0** blind ablation, **E6** zero-shot VLMs | Write the annotation guideline; run the translation audit (doc 06-A) |
| **W2** | Dec 1–7 | **E2** α sweep, **E4** rationale length, **E5** English control | Draft Related Work + Dataset section |
| **Gate 1** | Dec 7 | Does best-α Explanation reach ≥ w/o-R − 1 pt on RAD and SLAKE? **Yes** → continue with AARL as main. **No** → adopt the fallback framing (doc 04) and spend W3 on E3 + human eval anyway. | |
| **W3** | Dec 8–14 | **E3b/E3c** encoder + warm-up, AARL-full 3 seeds; Path runs | Prepare human-eval sheets (needs vanilla outputs, which are ready) |
| **W4** | Dec 15–21 | **E8** open-end final, **E7** analyses, Tier 2 if quota remains | **Human rationale evaluation** (doc 06-B) |
| **W5** | Dec 22–28 | Re-runs / fixes only | Write Method, Experiments, Results; make figures |
| **W6** | Dec 29–Jan 4 | – | Discussion, limitations, abstract; internal review by a co-author |
| **W7** | Jan 5–11 | – | Polish, release code + data (GitHub + HF Datasets / Kaggle with DOI via Zenodo), cover letter |
| **Submit** | ~Jan 12–15 | | |

Buffer: W5 is the slack. If Path runs slip, report Path at 1 seed and say so.

## Target journals (Q1, IF ~5+)

Impact factors are approximate (recent JCR) and should be checked on the journal site before choosing.

| Journal | Approx. IF | Fit | Notes |
|---|---|---|---|
| **Computers in Biology and Medicine** (Elsevier) | ~7 | ★★★ | Accepts methods + datasets in medical AI; relatively fast first decision. **Recommended first choice.** |
| **Artificial Intelligence in Medicine** (Elsevier) | ~6–7 | ★★★ | Values explainability and clinical evaluation. The human eval matters a lot here. |
| IEEE Journal of Biomedical and Health Informatics | ~6–7 | ★★ | Prefers stronger method novelty |
| Journal of Biomedical Informatics | ~4–5 | ★★ | Strong for the analysis/evaluation angle |
| Expert Systems with Applications | ~7–8 | ★★ | Broad, high volume, accepts applied work |
| Language Resources and Evaluation (Springer) | Q1 (lower IF) | ★★ | Best home if the paper ends up resource + analysis only |

Avoid submitting to a journal with very long review times if a January submission must produce a decision for a deadline. Check each journal's "time to first decision" on its home page.

## Paper outline (~10–12 journal pages)

1. **Introduction**: low-resource clinical AI, the need for small explainable models, the MedThink gap, and the reversal finding as the hook. 5 contributions (C1–C5).
2. **Related Work**: Med-VQA; rationale/CoT supervision; multilingual and Bangla medical AI (cite both Bangla Med-VQA papers explicitly and differentiate).
3. **BnMedThink Benchmark**: construction (LLM translation → full human check), terminology policy, quality statistics, splits, comparison table to existing Bangla VQA datasets.
4. **Method**
   - 4.1 MedThink-BN baseline (BanglaT5 + gated fusion)
   - 4.2 Why rationales hurt: a loss-share analysis (the answer-token share formula)
   - 4.3 AARL: segment-balanced loss, medical visual grounding, gate warm-up
5. **Experimental Setup**: protocol, seeds, metrics (Bengali-aware), statistical tests, compute (single T4, emphasised).
6. **Results**
   - 6.1 The reversal under a controlled EN–BN comparison
   - 6.2 Main results vs zero-shot VLMs
   - 6.3 Ablations
   - 6.4 Open-end
   - 6.5 Visual grounding analysis
   - 6.6 Human evaluation
   - 6.7 Error analysis by organ / question type
7. **Discussion**: implications for building explainable Med-VQA in low-resource languages, i.e. practical guidelines (weight the answer, ground vision, shorten rationales).
8. **Limitations**: translated (not native) questions; VQA-RAD image overlap; small human-eval sample; single backbone family; no clinical deployment study.
9. **Conclusion**; data/code availability statement; ethics statement.

## Things to state explicitly in the paper (deviations and honesty)

- lr 1e-4 vs 5e-4, and why (T5 v1.1 instability).
- Closed-gate init, and its ablation.
- Epoch caps / early stopping vs the paper's fixed epochs (the paper-protocol replication goes in the appendix).
- R-Path split counts differ from MedThink's (9428/3257 vs 9806/3391).
- RAD test-image overlap.
