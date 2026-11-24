# 03 — Issues Reviewers Will Raise, and Required Fixes

Ordered by severity. "Must" items will get a Q1 paper rejected if missing.

| # | Issue | Severity | Fix | Cost |
|---|---|---|---|---|
| 1 | Comparison to English uses the **paper's numbers** (different code, lr, backbone, split) | **Must** | Run the English parallel data through *your* pipeline with UnifiedQA-T5-base (E5). If possible, also run mT5-base on EN and BN so the backbone is held constant. | 4–17 GPU-h |
| 2 | **Single seed**, no variance | **Must** | 3 seeds for main configs; report mean ± std; McNemar / paired bootstrap for key pairs | Built into the plan |
| 3 | No **image-blind ablation**; gate stays near init | **Must** | Inference with zeroed and shuffled image features (E0); report the gate distribution | < 1 GPU-h |
| 4 | Translation quality not quantified | **Must** (resource paper) | Report post-edit statistics: % items edited, HTER / chrF between raw LLM output and final text, error categories (terminology, omission, mistranslation, register). Add a double-annotated 300-item audit with agreement. | Annotator time |
| 5 | Open-end evaluation by n-gram overlap only | **Must** | Answer EM / token-F1 with a seen/unseen split; BERTScore with BanglaBERT for rationales; **expert human evaluation** (doc 06) | Small |
| 6 | No validation split; 150/300 epochs is far past convergence | High | Carve a 10% val split from train for RAD/SLAKE (grouped by image for SLAKE); early-stop on val accuracy (closed) / val answer-F1 (open). Use Path's existing val. This also cuts compute 3–5×. | Saves GPU-h |
| 7 | Two-Stage open-end `full/*` row not comparable | High | Report Two-Stage only in `ans/*` and rationale tables, or recompute `full` as stage-1 rationale + stage-2 answer | Trivial |
| 8 | Truncated SLAKE open-end runs (207/300, 209/300) | High | Moot under the new early-stopping protocol; rerun with it | In plan |
| 9 | No comparison to modern VLMs | High | Zero-shot Qwen2.5-VL-3B and MedGemma-4B (inference only, fits a T4) in Bangla and English (E6) | ~3–4 GPU-h |
| 10 | Generic DETR encoder | Medium | BiomedCLIP patch features (E3) | ~1 h extraction |
| 11 | RAD test images 97% overlap with train | Medium | Disclose; optionally report results on the non-overlapping subset | Trivial |
| 12 | lr 1e-4 vs paper 5e-4 | Low | Already documented; add a one-line lr sanity sweep for the English control | Small |
| 13 | Gate init −3 may suppress vision | Medium | Ablate bias ∈ {−3, −1} + 1-epoch vision warm-up (E3b) | ~3 GPU-h |

## Protocol changes (adopt before any new run)

1. **Splits:** `val` = 10% of train (seeded, grouped by image for SLAKE/Path). Freeze the split files and publish them.
2. **Epoch caps with early stopping:** RAD 40, SLAKE 60, Path 12; patience 5 (RAD/SLAKE) / 3 (Path). Evaluate val every epoch for closed-end (fast: 1-token answers via choice scoring) and every 2 epochs for open-end.
3. **Seeds:** {42, 43, 44} for every number that goes into a main table.
4. **Decoding:** greedy (as now). Also report **choice scoring** for closed-end: compare log p("উত্তর: হ্যাঁ") vs log p("উত্তর: না"). It removes the "unresolved" failure mode and is a cleaner accuracy measure. Inference only.
5. **Statistics:** McNemar for closed-end pairs, paired bootstrap (10k resamples) for open-end metrics, Holm correction across the 3 datasets.
6. Keep the old 24-run grid as a **"paper-protocol replication"** appendix table. It is still valuable.

## Code changes needed (small)

| Change | Where | Size |
|---|---|---|
| Segment-weighted loss (answer vs rationale) | `train_model`, around `medthink_bn.py:1209`: compute per-token CE with `reduction="none"` and build an answer/rationale mask from the `MARK_ANS`/`MARK_RAT` token positions | ~40 lines |
| Choice-scoring evaluation | new function next to `generate_predictions` | ~30 lines |
| Image ablation (`zero`, `shuffle`) flag | `MedThinkDataset.__getitem__` / eval only | ~10 lines |
| Pluggable visual features (`patch_dim=768, patch_num=197` for BiomedCLIP) | Config + new extraction notebook | Config already parameterised |
| Validation split + early stopping | `train_model` | ~40 lines |
| Seed in `run_name()` | `Config.run_name` | 1 line |

Sketch of the loss (Explanation target `উত্তর: A কারণ: R`):

```python
# logits: (B, T, V); labels: (B, T) with -100 padding
tok_loss = F.cross_entropy(logits.transpose(1, 2), labels,
                           ignore_index=-100, reduction="none")      # (B, T)
valid = labels.ne(-100)
is_ans = answer_mask & valid          # tokens of "উত্তর: A" (+ eos for w/o-R)
is_rat = valid & ~is_ans
l_ans = (tok_loss * is_ans).sum(1) / is_ans.sum(1).clamp(min=1)
l_rat = (tok_loss * is_rat).sum(1) / is_rat.sum(1).clamp(min=1)
loss = (alpha * l_ans + (1 - alpha) * l_rat).mean()
```

Build `answer_mask` in the dataset: tokenize the answer segment separately and mark its length (for Explanation it is a prefix; for Reasoning it is a suffix).
