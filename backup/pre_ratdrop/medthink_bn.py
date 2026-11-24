"""
medthink_bn.py
==============
Shared library for reproducing MedThink (Gai et al., Findings of NAACL 2025)
on Bengali Med-VQA data: R-RAD-Bengali, R-SLAKE-Bangla, R-PathVQA-Bengali.

Faithful to the paper except for one deliberate, necessary substitution:

    UnifiedQA / T5-base  ->  csebuetnlp/banglat5   (or google/mt5-base)

The paper's TextualEncoder/Decoder is UnifiedQA (T5-base), whose SentencePiece
vocabulary contains no Bengali codepoints; Bengali input degenerates to <unk>.
BanglaT5 (247M) is Bengali-native and the closest match to the paper's 223M
budget, so its "0.2B parameters" claim still holds. Set model_name to
"google/mt5-base" for the multilingual 580M alternative.

Both are T5 v1.1 (gated-gelu, untied embeddings) and diverge to NaN at the
paper's lr of 5e-4. Trained here at 1e-4 with the visual gate initialised
closed; see MedThinkMultimodalGeneration._init_visual.

Everything else follows the paper:
  * VisualEncoder      : DETR (Carion et al. 2020) -> (100, 256) features
  * Cross-attention    : Eq. 2, softmax(QK^T/sqrt(d))V with Q from text, K/V from image
  * Gated fusion       : Eq. 3-4, lambda = sigmoid(Wl*F_T + Wv*H_attn)
  * Loss               : Eq. 6, token-level NLL
  * Strategies         : without_R, Explanation, Reasoning, Two-Stage Reasoning

Metrics are Bengali-aware and implemented from scratch (no English stemmers,
no whitespace-only tokenisation) so scores are meaningful on Bengali script.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DATASETS", "STRATEGIES", "Config",
    "bn_normalize", "bn_tokenize",
    "corpus_bleu", "rouge_scores", "exact_match", "token_f1", "chrf",
    "evaluate_open_end", "evaluate_open_end_dual", "evaluate_closed_end", "snap_to_choice",
    "load_split", "unify_records", "build_image_index", "resolve_image",
    "build_input_text", "build_target_text", "parse_answer",
    "MedThinkDataset", "collate_factory",
    "MedThinkMultimodalGeneration", "BanglaT5MultimodalGeneration", "load_model",
    "set_seed", "count_parameters", "diagnose_forward",
    "measure_gate", "majority_baseline", "constant_answer_baseline",
    "degenerate_rate", "record_uid", "find_checkpoint", "ensure_writable_ckpt",
    "train_model", "generate_predictions", "save_results",
    "split_train_val", "cap_records", "answer_segment_mask", "segment_loss",
    "score_choices", "evaluate_val", "image_ablation", "ANSWER_FIRST",
]

# --------------------------------------------------------------------------- #
# 0. Dataset registry
# --------------------------------------------------------------------------- #

# Filenames differ per dataset (the Bengali release is not internally uniform).
DATASETS: Dict[str, Dict[str, Any]] = {
    "rad": {
        "dir": "R-RAD-Bengali",
        "image_field": "image",
        "splits": {
            "closed-end": {"train": "trainset_bengali.json", "test": "testset_bengali.json"},
            "open-end": {"train": "trainset_bengali.json", "test": "testset_bengali.json"},
        },
        # open-end answers are int indices into a 1-element `choices` list
        "open_answer_mode": "index",
        "epochs": 150,          # paper Sec. 5.1
        # v2 protocol: cap + early stopping on a val split carved from train.
        # 97% of R-RAD test images also occur in train, so the val split is
        # NOT grouped by image here -- grouping would make val unlike test.
        "epochs_v2": 40,
        "patience_v2": 15,
        "val_group_by_image": False,
    },
    "slake": {
        "dir": "R-SLAKE-Bangla",
        "image_field": "img_name",
        "splits": {
            "closed-end": {"train": "train_bengali.json", "test": "test_bengali.json"},
            "open-end": {"train": "train_bengali.json", "test": "test_bengali.json"},
        },
        "open_answer_mode": "index",
        "epochs": 300,          # paper Sec. 5.1
        # SLAKE train/test share no images, so val must not either.
        "epochs_v2": 60,
        "patience_v2": 15,
        "val_group_by_image": True,
    },
    "path": {
        "dir": "R-PathVQA-Bengali",
        "image_field": "image",
        "splits": {
            "closed-end": {
                "train": "trainset_bengali.json",
                "val": "valset_bengali.json",
                "test": "testset_bengali.json",
            },
            "open-end": {
                "train": "trainset_bengali.json",
                "val": "valset_bengali.json",
                "test": "testset_bengali.json",
            },
        },
        # open-end answers are raw strings; `choices` is a vestigial ["হ্যাঁ","না"]
        # left over from the closed-end template and MUST be ignored.
        "open_answer_mode": "string",
        "epochs": 50,           # paper Sec. 5.1
        # Ships its own val split; only used if that split is missing.
        "epochs_v2": 12,
        "patience_v2": 5,
        "val_group_by_image": True,
    },
}

STRATEGIES = [
    "without_R",                # baseline: answer only
    "Explanation",              # A then R
    "Reasoning",                # R then A
    "First-Stage_Reasoning",    # (Q,I) -> R
    "Second-Stage_Reasoning",   # (Q,I,R) -> A
]

# Bengali surface markers. Using Bengali rather than the paper's English
# "The answer is (X)" keeps the target distribution inside the backbone's
# Bengali pretraining domain; the structural role is identical.
MARK_Q = "প্রশ্ন:"
MARK_OPT = "বিকল্প:"
MARK_ANS = "উত্তর:"
MARK_RAT = "কারণ:"
OPTION_LETTERS = ["ক", "খ", "গ", "ঘ", "ঙ", "চ"]

# Per-language surface markers. "bn" (above) is the study language; "en" is the
# same-pipeline English control (experiment E5), trained on the original
# MedThink English JSONs. Switch with set_language(); never edit the globals.
LANG_MARKERS: Dict[str, Dict[str, Any]] = {
    "bn": dict(MARK_Q=MARK_Q, MARK_OPT=MARK_OPT, MARK_ANS=MARK_ANS, MARK_RAT=MARK_RAT,
               OPTION_LETTERS=list(OPTION_LETTERS)),
    "en": dict(MARK_Q="Question:", MARK_OPT="Options:", MARK_ANS="Answer:", MARK_RAT="Because:",
               OPTION_LETTERS=["A", "B", "C", "D", "E", "F"]),
}
CURRENT_LANG = "bn"

# English originals (folder `Medthink-Dataset/` inside towhaelahi/medthink-dataset).
# Same record ids, image names and answer indices as the Bengali releases.
EN_DATASETS: Dict[str, Dict[str, Any]] = {
    "rad":   {"dir": "R-RAD",     "files": {"train": "trainset.json", "test": "testset.json"}},
    "slake": {"dir": "R-SLAKE",   "files": {"train": "train.json", "test": "test.json"}},
    "path":  {"dir": "R-PathVQA", "files": {"train": "trainset.json", "val": "valset.json",
                                            "test": "testset.json"}},
}


@dataclass
class Config:
    """Every knob for one training run. Serialised alongside results."""
    dataset: str = "rad"                       # rad | slake | path
    subset: str = "closed-end"                 # closed-end | open-end
    strategy: str = "Explanation"              # see STRATEGIES

    # BanglaT5 (247M) -- Bengali-native and closest to the paper's 223M, so the
    # paper's "0.2B parameters" comparison survives. Alternative:
    # "google/mt5-base" (580M, multilingual; then use batch_size=2,
    # grad_accum=16, since it needs ~9GB before activations).
    # Both are T5 v1.1 (gated-gelu, untied embeddings), an unstable family, so
    # the low lr and closed-gate init below are load-bearing, not cosmetic.
    model_name: str = "csebuetnlp/banglat5"
    patch_dim: int = 256                       # DETR hidden size
    patch_num: int = 100                       # DETR object queries

    # With dynamic padding these are truncation caps only, not the padded
    # width, so generous values cost nothing. Real inputs are 11-78 tokens and
    # targets reach 318, so nothing is actually truncated.
    source_len: int = 512
    target_len: int = 320                      # raised from 64: Bengali rationales
    #                                            average ~75 words (~150+ tokens)

    # The paper uses 5e-4, tuned for UnifiedQA (T5 v1.0). BanglaT5 is T5 v1.1
    # (feed_forward_proj="gated-gelu", tie_word_embeddings=False), which is
    # markedly less stable: 5e-4 drives it to NaN within a few steps. 1e-4 is
    # the standard fine-tuning rate for the v1.1 family. Documented deviation.
    lr: float = 1e-4
    lr_stage2: float = 5e-5                    # paper: second stage
    epochs: Optional[int] = None               # None -> paper value for dataset
    epochs_stage2: int = 20                    # paper Sec. 5.1
    # 8 x 4 = the paper's effective batch of 32. Dynamic padding cut real
    # sequence widths from 512 to ~30, so a larger micro-batch now fits and
    # keeps the GPU better occupied. For mT5-base (580M) use 4 x 8.
    batch_size: int = 8
    grad_accum: int = 4                        # effective batch 32 (paper)
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.06
    label_smoothing: float = 0.0
    max_grad_norm: float = 1.0

    # DataLoader workers. Kaggle is Linux and forks cheaply; Windows spawns,
    # which needs an `if __name__ == "__main__"` guard the notebooks cannot
    # provide, so default to 0 there.
    num_workers: int = 0 if os.name == "nt" else 2

    num_beams: int = 1
    seed: int = 42
    fp16: bool = False                         # T5 overflows in fp16; keep off
    quick_run: bool = False                    # tiny subset smoke test

    # "paper": fixed paper epochs, no validation -- reproduces the original
    #          24-run grid exactly.
    # "v2":    epoch cap (DATASETS[..]["epochs_v2"]) with early stopping on a
    #          held-out val split. Every v2 number must be compared only with
    #          other v2 numbers.
    protocol: str = "paper"
    val_frac: float = 0.1                      # carved from train when no val split exists
    val_max: int = 500                         # cap on val items scored per epoch
    patience: Optional[int] = None             # None -> protocol default
    # Early stopping may not fire before this epoch (None -> 50% of the v2
    # cap; was 25% with patience 5/5/3 until 2026-11-21, which stopped runs
    # while noisy val accuracy was still rising -- see docs/10). Without it, the flat val accuracy while the model is still
    # learning the answer format ends runs prematurely (seen in the first
    # real-GPU sanity run: stopped at epoch 10 with train loss still falling).
    min_epochs: Optional[int] = None
    eval_every: int = 1                        # validate every N epochs

    # Segment-balanced loss: alpha * L_answer + (1 - alpha) * L_rationale,
    # each length-normalised per example. None keeps the paper's token-mean
    # NLL (Eq. 6), under which a 1-2 token answer gets ~2% of the gradient
    # next to a ~150-token rationale.
    ans_weight: Optional[float] = None

    # Visual pathway. -3.0 (lambda ~0.047) is what the 24-run grid used.
    gate_bias_init: float = -3.0
    # Freeze the text backbone for the first N epochs so the randomly
    # initialised visual pathway learns before competing with pretrained text.
    vision_warmup_epochs: int = 0
    lang: str = "bn"                           # bn | en (E5 English control; see set_language)

    data_root: str = "/kaggle/input"
    feature_dir: str = "/kaggle/input/medthink-bn-features"
    output_dir: str = "/kaggle/working"

    def resolved_epochs(self) -> int:
        if self.epochs is not None:
            return self.epochs
        if self.protocol == "v2":
            return DATASETS[self.dataset]["epochs_v2"]
        return DATASETS[self.dataset]["epochs"]

    def resolved_patience(self) -> int:
        """Epochs without val improvement before stopping. 0 disables."""
        if self.patience is not None:
            return self.patience
        if self.protocol == "v2":
            return DATASETS[self.dataset]["patience_v2"]
        return 0

    def resolved_min_epochs(self) -> int:
        """First epoch at which early stopping is allowed to stop the run."""
        if self.min_epochs is not None:
            return self.min_epochs
        return max(1, round(0.5 * self.resolved_epochs())) if self.protocol == "v2" else 0

    def model_tag(self) -> str:
        """
        Short backbone id, e.g. 'mt5-base' or 'banglat5'.

        Must be filename-safe: it goes into every results filename. A local
        checkpoint directory (offline Kaggle runs point `model_name` at one)
        would otherwise embed path separators and produce an unopenable path.
        """
        tag = self.model_name.replace("\\", "/").rstrip("/").split("/")[-1]
        return re.sub(r"[^A-Za-z0-9._-]+", "-", tag) or "model"

    def run_name(self) -> str:
        # The backbone is part of the identity: without it, an mT5 run and a
        # BanglaT5 run of the same dataset/strategy would write to the same
        # metrics file and silently overwrite each other.
        name = f"{self.dataset}_{self.subset}_{self.strategy}_{self.model_tag()}"
        # Non-default settings are appended, so paper-protocol runs keep the
        # names (and resumable checkpoints) of the original grid, while seeds
        # and ablations never overwrite each other.
        if self.protocol != "paper":
            name += f"_{self.protocol}_s{self.seed}"
        elif self.seed != 42:
            name += f"_s{self.seed}"
        if self.ans_weight is not None:
            name += f"_aw{self.ans_weight:g}"
        if self.gate_bias_init != -3.0:
            name += f"_gb{self.gate_bias_init:g}"
        if self.vision_warmup_epochs:
            name += f"_vw{self.vision_warmup_epochs}"
        # Non-DETR visual features (e.g. BiomedCLIP 197 x 768) get their own
        # namespace; DETR (100 x 256) names stay unchanged.
        if (self.patch_num, self.patch_dim) != (100, 256):
            name += f"_f{self.patch_num}x{self.patch_dim}"
        if self.lang != "bn":
            name += f"_{self.lang}"
        # A quick_run is a diagnostic, not a result. Keeping it in a separate
        # namespace means re-running the sanity check after a real run cannot
        # overwrite hours of work with a 5-minute smoke test.
        return name + "_SANITY" if self.quick_run else name

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# 1. Bengali normalisation + tokenisation
# --------------------------------------------------------------------------- #

# csebuetnlp's normalizer is the reference Bengali normaliser (and what
# BanglaT5 expects). Optional here: if the package
# is unavailable (offline Kaggle session) we fall back to an NFKC-based
# normaliser that handles the cases that actually matter for this corpus.
try:  # pragma: no cover - environment dependent
    from normalizer import normalize as _csebuetnlp_normalize  # type: ignore
    _HAS_CSEBUETNLP = True
except Exception:  # pragma: no cover
    _csebuetnlp_normalize = None
    _HAS_CSEBUETNLP = False

# Bengali digits -> ASCII, so "২" and "2" compare equal.
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Characters that are visually identical but differently encoded.
_BN_FIXES = {
    "ড়": "ড়",   # RRA  -> DDA + nukta
    "ঢ়": "ঢ়",   # RHA  -> DHA + nukta
    "য়": "য়",   # YYA  -> YA  + nukta
    "‌": "",               # ZWNJ
    "‍": "",               # ZWJ
    "৤": "।", "৥": "।",
}

_PUNCT_RE = re.compile(r"[।,;:!?\"'`()\[\]{}<>/\\|@#$%^&*_+=~—–\-\.]+")
_WS_RE = re.compile(r"\s+")
# Bengali block + ASCII alphanumerics: medical text is code-mixed (DWI, MRI...).
_TOKEN_RE = re.compile(r"[ঀ-৿]+|[A-Za-z]+|[0-9]+")


def bn_normalize(text: str, use_csebuetnlp: bool = True) -> str:
    """Normalise Bengali text for both model input and metric computation."""
    if text is None:
        return ""
    text = str(text)
    if use_csebuetnlp and _HAS_CSEBUETNLP:
        try:
            text = _csebuetnlp_normalize(text)
        except Exception:
            pass
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _BN_FIXES.items():
        text = text.replace(bad, good)
    text = text.translate(_BN_DIGITS)
    return _WS_RE.sub(" ", text).strip()


def bn_tokenize(text: str, lower: bool = True) -> List[str]:
    """
    Tokenise Bengali for BLEU/ROUGE.

    Whitespace splitting leaves punctuation glued to words ("ফুসফুস।" vs
    "ফুসফুস"), which silently deflates n-gram overlap. English tokenisers are
    worse: they stem and lowercase against a Latin alphabet. This splits on the
    Bengali Unicode block plus Latin/digit runs, which is what the scores in
    the paper's Table 3 are actually meant to measure.
    """
    text = bn_normalize(text)
    if lower:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def strip_punct(text: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text)).strip()


# --------------------------------------------------------------------------- #
# 2. Metrics (pure Python, Bengali-safe)
# --------------------------------------------------------------------------- #

def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Cumulative corpus BLEU-1..BLEU-n with brevity penalty.

    Uses the standard clipped-precision formulation. A tiny epsilon replaces a
    zero precision so that BLEU-4 does not collapse to exactly 0 on short
    single-phrase answers (common in open-end Med-VQA), matching the smoothing
    that BLEU implementations conventionally apply.
    """
    assert len(hypotheses) == len(references)
    clipped = [0] * (max_n + 1)
    total = [0] * (max_n + 1)
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        h = bn_tokenize(hyp)
        r = bn_tokenize(ref)
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            h_ng = _ngrams(h, n)
            r_ng = _ngrams(r, n)
            total[n] += max(sum(h_ng.values()), 0)
            clipped[n] += sum(min(c, r_ng[g]) for g, c in h_ng.items())

    bp = 1.0 if hyp_len > ref_len else (math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0)

    out: Dict[str, float] = {}
    log_sum = 0.0
    for n in range(1, max_n + 1):
        if total[n] == 0:
            # No n-grams of this order exist anywhere in the corpus -- e.g. a
            # constant one-token baseline has no bigrams. BLEU-n is then
            # genuinely UNDEFINED, not zero and not perfect. Reporting 1.0 here
            # (an earlier attempt) makes BLEU-n rise with n, so BLEU-4 exceeds
            # BLEU-1 on short answers, which is nonsense. NaN is the honest
            # answer, and every higher order is undefined too.
            for m in range(n, max_n + 1):
                out[f"BLEU-{m}"] = float("nan")
            break
        if clipped[n] == 0:
            p = 1e-9        # genuine total miss -> smoothed, not division by zero
        else:
            p = clipped[n] / total[n]
        log_sum += math.log(p)
        out[f"BLEU-{n}"] = 100.0 * bp * math.exp(log_sum / n)
    return out


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_scores(hypotheses: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """ROUGE-1/2/L F1, averaged over examples, on Bengali tokens."""
    r1 = r2 = rl = 0.0
    n = max(len(hypotheses), 1)

    for hyp, ref in zip(hypotheses, references):
        h, r = bn_tokenize(hyp), bn_tokenize(ref)

        for k, acc in ((1, "r1"), (2, "r2")):
            h_ng, r_ng = _ngrams(h, k), _ngrams(r, k)
            overlap = sum(min(c, r_ng[g]) for g, c in h_ng.items())
            hs, rs = sum(h_ng.values()), sum(r_ng.values())
            p = overlap / hs if hs else 0.0
            rc = overlap / rs if rs else 0.0
            f = 2 * p * rc / (p + rc) if (p + rc) else 0.0
            if acc == "r1":
                r1 += f
            else:
                r2 += f

        l = _lcs_len(h, r)
        p = l / len(h) if h else 0.0
        rc = l / len(r) if r else 0.0
        rl += 2 * p * rc / (p + rc) if (p + rc) else 0.0

    return {"ROUGE-1": 100.0 * r1 / n, "ROUGE-2": 100.0 * r2 / n, "ROUGE-L": 100.0 * rl / n}


def exact_match(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    hits = sum(
        1
        for h, r in zip(hypotheses, references)
        if strip_punct(bn_normalize(h)).lower() == strip_punct(bn_normalize(r)).lower()
    )
    return 100.0 * hits / max(len(hypotheses), 1)


def token_f1(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    tot = 0.0
    for h, r in zip(hypotheses, references):
        ht, rt = Counter(bn_tokenize(h)), Counter(bn_tokenize(r))
        overlap = sum(min(c, rt[g]) for g, c in ht.items())
        hs, rs = sum(ht.values()), sum(rt.values())
        p = overlap / hs if hs else 0.0
        rc = overlap / rs if rs else 0.0
        tot += 2 * p * rc / (p + rc) if (p + rc) else 0.0
    return 100.0 * tot / max(len(hypotheses), 1)


def chrf(hypotheses: Sequence[str], references: Sequence[str], n: int = 6, beta: float = 2.0) -> float:
    """chrF++-style character n-gram F-score. Suits agglutinative Bengali."""
    tot = 0.0
    for hyp, ref in zip(hypotheses, references):
        h = bn_normalize(hyp).replace(" ", "")
        r = bn_normalize(ref).replace(" ", "")
        fs = []
        for k in range(1, n + 1):
            hg, rg = _ngrams(h, k), _ngrams(r, k)
            overlap = sum(min(c, rg[g]) for g, c in hg.items())
            hs, rs = sum(hg.values()), sum(rg.values())
            if hs == 0 or rs == 0:
                continue
            p, rc = overlap / hs, overlap / rs
            if p + rc == 0:
                fs.append(0.0)
            else:
                b2 = beta ** 2
                fs.append((1 + b2) * p * rc / (b2 * p + rc))
        tot += sum(fs) / len(fs) if fs else 0.0
    return 100.0 * tot / max(len(hypotheses), 1)


def evaluate_open_end(hypotheses: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """Full open-end metric suite (paper Table 3 + correctness diagnostics)."""
    out: Dict[str, float] = {}
    out.update(rouge_scores(hypotheses, references))
    out.update(corpus_bleu(hypotheses, references))
    out["ExactMatch"] = exact_match(hypotheses, references)
    out["TokenF1"] = token_f1(hypotheses, references)
    out["chrF"] = chrf(hypotheses, references)
    return out


def evaluate_open_end_dual(
    full_preds: Sequence[str],
    full_refs: Sequence[str],
    answer_preds: Sequence[str],
    answer_refs: Sequence[str],
) -> Dict[str, float]:
    """
    Open-end scoring at two granularities.

    `full_*`   : whole generation vs whole target (answer + rationale).
    `answer_*` : parsed answer span vs gold answer only.

    Both are reported because they measure different things and the paper is
    ambiguous about which it used. Table 3 must be the *full* variant: R-SLAKE
    open-end answers average 1-2 tokens, which admits no 4-grams at all, so the
    reported BLEU-4 of 10.3 cannot come from answer-only scoring. The
    answer-only numbers are the clinically meaningful ones -- a model can score
    well on rationale overlap while getting the diagnosis wrong.
    """
    out: Dict[str, float] = {}
    for k, v in evaluate_open_end(full_preds, full_refs).items():
        out[f"full/{k}"] = v
    for k, v in evaluate_open_end(answer_preds, answer_refs).items():
        out[f"ans/{k}"] = v
    return out


def evaluate_closed_end(pred_texts: Sequence[str], gold_texts: Sequence[str],
                        choices: Sequence[Sequence[str]]) -> Dict[str, float]:
    """
    Accuracy for closed-end questions (paper Table 2).

    Generation is free-form, so the prediction is snapped to the nearest option
    before scoring: exact match first, then substring, then token-overlap.
    """
    correct = 0
    unresolved = 0
    for pred, gold, opts in zip(pred_texts, gold_texts, choices):
        choice = snap_to_choice(pred, opts)
        if choice is None:
            unresolved += 1
        elif strip_punct(bn_normalize(choice)).lower() == strip_punct(bn_normalize(gold)).lower():
            correct += 1
    n = max(len(pred_texts), 1)
    return {
        "Accuracy": 100.0 * correct / n,
        "Unresolved": 100.0 * unresolved / n,
    }


def snap_to_choice(pred: str, options: Sequence[str]) -> Optional[str]:
    """Map a free-form generation onto one of the answer options."""
    p = strip_punct(bn_normalize(pred)).lower()
    norm_opts = [strip_punct(bn_normalize(o)).lower() for o in options]

    for o, no in zip(options, norm_opts):
        if p == no:
            return o
    # Prefer the option appearing earliest in the generation.
    best, best_pos = None, len(p) + 1
    for o, no in zip(options, norm_opts):
        if no and no in p:
            pos = p.index(no)
            if pos < best_pos:
                best, best_pos = o, pos
    if best is not None:
        return best

    pt = Counter(bn_tokenize(p))
    best, best_score = None, 0.0
    for o, no in zip(options, norm_opts):
        ot = Counter(bn_tokenize(no))
        overlap = sum(min(c, ot[g]) for g, c in pt.items())
        denom = sum(ot.values())
        score = overlap / denom if denom else 0.0
        if score > best_score:
            best, best_score = o, score
    return best if best_score > 0 else None


# --------------------------------------------------------------------------- #
# 3. Data loading and schema unification
# --------------------------------------------------------------------------- #

def load_split(data_root: str, dataset: str, subset: str, split: str) -> Dict[str, dict]:
    """Load one raw Bengali JSON file. Returns {id: record}."""
    meta = DATASETS[dataset]
    fname = meta["splits"][subset].get(split)
    if fname is None:
        raise KeyError(f"{dataset}/{subset} has no '{split}' split")

    root = Path(data_root)
    candidates = list(root.glob(f"**/{meta['dir']}/{subset}/{fname}"))
    if not candidates:
        candidates = list(root.glob(f"**/{subset}/{fname}"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {meta['dir']}/{subset}/{fname} under {data_root}. "
            f"Attach the Bengali dataset to the notebook."
        )
    with open(candidates[0], encoding="utf-8") as fh:
        return json.load(fh)


def unify_records(raw: Dict[str, dict], dataset: str, subset: str, split: str) -> List[dict]:
    """
    Flatten the three inconsistent schemas into one record type.

    The three releases disagree in ways that silently corrupt results if
    ignored:
      * R-RAD / R-SLAKE open-end : `answer` is an int index into `choices`
      * R-PathVQA open-end       : `answer` is the answer STRING, and `choices`
                                   is a leftover ["হ্যাঁ","না"] that is wrong
      * R-SLAKE                  : image key is `img_name`, not `image`
    """
    meta = DATASETS[dataset]
    img_field = meta["image_field"]
    open_mode = meta["open_answer_mode"]
    out: List[dict] = []
    skipped = 0

    for qid, rec in raw.items():
        question = bn_normalize(rec.get("question", ""))
        choices = [bn_normalize(c) for c in rec.get("choices", []) or []]
        ans = rec.get("answer")

        if subset == "closed-end":
            if not isinstance(ans, int) or not choices or ans >= len(choices):
                skipped += 1
                continue
            answer_text = choices[ans]
            options = choices
        else:
            if open_mode == "string":
                if not isinstance(ans, str) or not ans.strip():
                    skipped += 1
                    continue
                answer_text = bn_normalize(ans)
                options = []          # vestigial choices deliberately dropped
            else:
                if not isinstance(ans, int) or not choices or ans >= len(choices):
                    skipped += 1
                    continue
                answer_text = choices[ans]
                options = []

        solution = bn_normalize(rec.get("solution", "") or "")
        image_key = rec.get(img_field) or rec.get("image") or rec.get("img_name") or ""

        out.append({
            "qid": str(qid),
            "dataset": dataset,
            "subset": subset,
            "split": split,
            "question": question,
            "answer": answer_text,
            "options": options,
            "solution": solution,
            "image_key": str(image_key).replace("\\", "/"),
            "answer_type": rec.get("answer_type", ""),
            "image_organ": rec.get("image_organ", "") or rec.get("location", ""),
            "content_type": rec.get("content_type", "") or rec.get("question_type", ""),
        })

    if skipped:
        print(f"  [unify] {dataset}/{subset}/{split}: skipped {skipped} malformed record(s)")
    return out


def split_train_val(records: List[dict], dataset: str, frac: float = 0.1,
                    seed: int = 0) -> Tuple[List[dict], List[dict]]:
    """
    Carve a validation split out of train for datasets that ship none.

    The seed is deliberately NOT the training seed: every run, seed and
    strategy must select checkpoints on the identical val set, or the seed
    variance being reported would include val-set variance too.

    Grouped by image where train and test share no images (R-SLAKE), so val
    mimics test; plain random where they do (R-RAD, 97% overlap).
    """
    group = DATASETS[dataset].get("val_group_by_image", True)
    keys = sorted({(r["image_key"] if group else record_uid(r)) for r in records})
    rng = random.Random(seed)
    rng.shuffle(keys)
    held = set(keys[:max(1, int(round(frac * len(keys))))])

    def key(r):
        return r["image_key"] if group else record_uid(r)

    train = [r for r in records if key(r) not in held]
    val = [dict(r, split="val") for r in records if key(r) in held]
    return train, val


def cap_records(records: List[dict], n: Optional[int], seed: int = 0) -> List[dict]:
    """Deterministic subsample, identical across runs (seed is not cfg.seed)."""
    if not n or n >= len(records):
        return records
    idx = sorted(random.Random(seed).sample(range(len(records)), n))
    return [records[i] for i in idx]


# --------------------------------------------------------------------------- #
# 4. Image resolution
# --------------------------------------------------------------------------- #

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_image_index(search_roots: Iterable[str]) -> Dict[str, str]:
    """
    Index every image under the given roots by several keys, so that whichever
    public image dataset is attached, lookups still resolve.

    Keys registered per file:
      * basename                     -> synpic54610.jpg
      * parent/basename              -> xmlab1/source.jpg   (SLAKE)
      * grandparent/parent/basename
    """
    index: Dict[str, str] = {}
    n = 0
    for root in search_roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for p in rp.rglob("*"):
            if p.suffix.lower() not in _IMG_EXT or not p.is_file():
                continue
            n += 1
            parts = p.parts
            keys = [p.name.lower()]
            if len(parts) >= 2:
                keys.append(f"{parts[-2]}/{p.name}".lower())
            if len(parts) >= 3:
                keys.append(f"{parts[-3]}/{parts[-2]}/{p.name}".lower())
            for k in keys:
                index.setdefault(k, str(p))
    print(f"  [image index] indexed {n} image files -> {len(index)} keys")
    return index


def resolve_image(image_key: str, index: Dict[str, str]) -> Optional[str]:
    """Resolve a record's image reference against the index."""
    if not image_key:
        return None
    k = image_key.lower().lstrip("./")
    if k in index:
        return index[k]
    parts = k.split("/")
    for cand in ("/".join(parts[-2:]), parts[-1]):
        if cand in index:
            return index[cand]
    # SLAKE masks/sources sometimes differ in extension only.
    stem = Path(parts[-1]).stem
    for ext in _IMG_EXT:
        cand = f"{stem}{ext}"
        if len(parts) >= 2:
            two = f"{parts[-2]}/{cand}"
            if two in index:
                return index[two]
        if cand in index:
            return index[cand]
    return None


# --------------------------------------------------------------------------- #
# 5. Prompt construction (the three strategies of paper Fig. 1c)
# --------------------------------------------------------------------------- #

def format_options(options: Sequence[str]) -> str:
    if not options:
        return ""
    parts = [f"({OPTION_LETTERS[i]}) {o}" for i, o in enumerate(options)]
    return f"{MARK_OPT} " + " ".join(parts)


def build_input_text(rec: dict, strategy: str, generated_rationale: str = "") -> str:
    """
    Model input.

    Only 'Second-Stage_Reasoning' consumes a rationale on the input side; it
    receives the rationale produced by the stage-1 model, never the gold one
    (using gold at inference would leak the answer).
    """
    parts = [f"{MARK_Q} {rec['question']}"]
    opts = format_options(rec.get("options", []))
    if opts:
        parts.append(opts)
    if strategy == "Second-Stage_Reasoning":
        parts.append(f"{MARK_RAT} {bn_normalize(generated_rationale)}")
    return "\n".join(parts)


def build_target_text(rec: dict, strategy: str) -> str:
    """Training target for each strategy."""
    a, r = rec["answer"], rec.get("solution", "")
    if strategy == "without_R":
        return f"{MARK_ANS} {a}"
    if strategy == "Explanation":                 # A then R
        return f"{MARK_ANS} {a} {MARK_RAT} {r}"
    if strategy == "Reasoning":                   # R then A
        return f"{MARK_RAT} {r} {MARK_ANS} {a}"
    if strategy == "First-Stage_Reasoning":       # rationale only
        return r
    if strategy == "Second-Stage_Reasoning":      # answer only, rationale given
        return f"{MARK_ANS} {a}"
    raise ValueError(f"unknown strategy: {strategy}")


_ANS_RE = re.compile(re.escape(MARK_ANS) + r"\s*(.*?)(?:\s*" + re.escape(MARK_RAT) + r"|$)", re.S)
_RAT_RE = re.compile(re.escape(MARK_RAT) + r"\s*(.*?)(?:\s*" + re.escape(MARK_ANS) + r"|$)", re.S)


def set_language(lang: str) -> None:
    """
    Switch the surface markers (and the regexes parsing them) to `lang`.

    Every builder, parser, answer-mask and choice-scoring function reads the
    module globals at call time, so one call before building datasets switches
    the whole pipeline. MedThinkDataset asserts it matches cfg.lang.
    """
    global MARK_Q, MARK_OPT, MARK_ANS, MARK_RAT, OPTION_LETTERS, CURRENT_LANG, _ANS_RE, _RAT_RE
    if lang not in LANG_MARKERS:
        raise ValueError(f"unknown lang {lang!r}; choose from {sorted(LANG_MARKERS)}")
    m = LANG_MARKERS[lang]
    MARK_Q, MARK_OPT, MARK_ANS, MARK_RAT = m["MARK_Q"], m["MARK_OPT"], m["MARK_ANS"], m["MARK_RAT"]
    OPTION_LETTERS = list(m["OPTION_LETTERS"])
    _ANS_RE = re.compile(re.escape(MARK_ANS) + r"\s*(.*?)(?:\s*" + re.escape(MARK_RAT) + r"|$)", re.S)
    _RAT_RE = re.compile(re.escape(MARK_RAT) + r"\s*(.*?)(?:\s*" + re.escape(MARK_ANS) + r"|$)", re.S)
    CURRENT_LANG = lang


def overlay_language(records: List[dict], data_root: str, lang: str, dataset: str,
                     subset: str, split: str) -> List[dict]:
    """
    Replace the Bengali text of manifest records with the parallel `lang` text.

    Keeps everything else (qid, image key, split membership, order), so the
    English control trains and is selected/tested on exactly the same items
    and image features as the Bengali runs. Fails loudly on any mismatch.
    """
    if lang == "bn":
        return records
    if lang != "en":
        raise ValueError(f"no overlay source for lang {lang!r}")
    meta = EN_DATASETS[dataset]
    fname = meta["files"][split]
    hits = sorted(Path(data_root).glob(f"**/{meta['dir']}/{subset}/{fname}"))
    if not hits:
        raise FileNotFoundError(f"{meta['dir']}/{subset}/{fname} not found under {data_root} "
                                "- attach towhaelahi/medthink-dataset")
    raw = json.load(open(hits[0], encoding="utf-8"))
    out, bad = [], []
    for rec in records:
        er = raw.get(rec["qid"])
        if er is None:
            bad.append((rec["qid"], "missing"))
            continue
        choices = er.get("choices") or []
        if isinstance(choices, str):
            import ast
            choices = ast.literal_eval(choices)
        choices = [bn_normalize(c) for c in choices]
        new = dict(rec)
        new["question"] = bn_normalize(er.get("question", ""))
        new["solution"] = bn_normalize(er.get("solution", "") or "")
        ans = er.get("answer")
        if subset == "closed-end" or (isinstance(ans, int) and DATASETS[dataset]["open_answer_mode"] == "index"):
            if not isinstance(ans, int) or ans >= len(choices):
                bad.append((rec["qid"], "answer index"))
                continue
            if rec.get("options") and rec["options"].index(rec["answer"]) != ans:
                bad.append((rec["qid"], "answer index differs from Bengali"))
                continue
            new["answer"] = choices[ans]
            new["options"] = choices if rec.get("options") else []
        else:
            new["answer"] = bn_normalize(str(ans))
        img = er.get(DATASETS[dataset]["image_field"]) or er.get("image") or er.get("img_name") or ""
        if str(img).replace("\\", "/").lower() != rec["image_key"].lower():
            bad.append((rec["qid"], "image differs"))
            continue
        out.append(new)
    if bad:
        raise ValueError(f"overlay {dataset}/{subset}/{split}: {len(bad)} mismatched records, e.g. {bad[:5]}")
    print(f"  [overlay] {dataset}/{subset}/{split}: {len(out)} records -> {lang} ({hits[0]})")
    return out


def parse_answer(text: str, strategy: str) -> Tuple[str, str]:
    """
    Split a generation into (answer, rationale).

    Works for both orderings because the markers are searched independently.
    If no marker survives generation, the whole string is treated as the
    answer for answer-bearing strategies.
    """
    if strategy == "First-Stage_Reasoning":
        return "", text.strip()
    a = _ANS_RE.search(text)
    r = _RAT_RE.search(text)
    answer = a.group(1).strip() if a else text.strip()
    rationale = r.group(1).strip() if r else ""
    return answer, rationale


def answer_segment_mask(tokenizer, rec: dict, strategy: str, labels: np.ndarray) -> np.ndarray:
    """
    1 for label positions belonging to the answer segment "উত্তর: A", else 0.

    The answer segment is tokenised on its own and its length marks a prefix
    (Explanation) or suffix (Reasoning) of the labels. SentencePiece splits on
    whitespace-prefixed pieces, so the segment tokenises identically in and
    out of context; `MedThinkDataset.check_answer_mask` decodes it to verify.
    """
    n = len(labels)
    mask = np.zeros(n, dtype=np.int64)
    if strategy in ("without_R", "Second-Stage_Reasoning"):
        mask[:] = 1                               # the whole target is the answer
        return mask
    if strategy == "First-Stage_Reasoning":
        return mask                               # no answer in the target

    seg = f"{MARK_ANS} {rec['answer']}"
    k = len(tokenizer(seg, add_special_tokens=False)["input_ids"])
    if strategy == "Explanation":                 # A first
        mask[:min(k, n)] = 1
    elif strategy == "Reasoning":                 # A last, then </s>
        # A target truncated at target_len has lost its answer; mark nothing
        # rather than mislabel rationale tokens as the answer.
        if n and labels[-1] == tokenizer.eos_token_id:
            mask[max(0, n - k - 1):] = 1
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return mask


# --------------------------------------------------------------------------- #
# 6. Torch dataset / collation
# --------------------------------------------------------------------------- #

class MedThinkDataset:
    """Pairs a unified record list with its cached DETR features."""

    def __init__(
        self,
        records: List[dict],
        tokenizer,
        cfg: Config,
        features: Optional[np.ndarray] = None,
        name_to_row: Optional[Dict[str, int]] = None,
        rationales: Optional[Dict[str, str]] = None,
    ):
        self.records = records
        self.tok = tokenizer
        self.cfg = cfg
        assert getattr(cfg, "lang", "bn") == CURRENT_LANG, (
            f"cfg.lang={cfg.lang!r} but markers are {CURRENT_LANG!r}: call M.set_language(cfg.lang) first")
        self.features = features
        self.name_to_row = name_to_row or {}
        self.rationales = rationales or {}
        self.missing = 0
        self.image_mode = "real"
        self._image_src = list(range(len(records)))

    def __len__(self) -> int:
        return len(self.records)

    def set_image_mode(self, mode: str = "real", seed: int = 0) -> float:
        """
        Image ablation for evaluation.

          real    : the record's own image
          zero    : all-zero features (image-blind)
          shuffle : another record's image, never the same image

        Returns the fraction of records left on their own image (0 for a
        clean shuffle) so the ablation can be checked, not assumed.
        """
        assert mode in ("real", "zero", "shuffle"), mode
        self.image_mode = mode
        n = len(self.records)
        self._image_src = list(range(n))
        if mode == "shuffle" and n > 1:
            rng = random.Random(seed)
            rng.shuffle(self._image_src)
            img = [r["image_key"] for r in self.records]
            # Repair collisions by swapping with a random other position.
            for _ in range(10):
                bad = [i for i in range(n) if img[self._image_src[i]] == img[i]]
                if not bad:
                    break
                for i in bad:
                    j = rng.randrange(n)
                    self._image_src[i], self._image_src[j] = self._image_src[j], self._image_src[i]
        same = sum(self.records[self._image_src[i]]["image_key"] == self.records[i]["image_key"]
                   for i in range(n))
        return same / max(n, 1) if mode == "shuffle" else float(mode == "real")

    def image_feature(self, rec: dict) -> np.ndarray:
        """DETR features for this record, or zeros when the image is absent."""
        zero = np.zeros((self.cfg.patch_num, self.cfg.patch_dim), dtype=np.float32)
        if self.features is None or self.image_mode == "zero":
            return zero
        row = self.name_to_row.get(rec["image_key"].lower())
        if row is None:
            self.missing += 1
            return zero
        feat = np.asarray(self.features[row], dtype=np.float32)
        # A single inf/nan from a corrupt image would poison the whole batch.
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        # Keyed by the globally unique id, not qid -- see record_uid().
        gen_rat = self.rationales.get(record_uid(rec), "")
        src = build_input_text(rec, self.cfg.strategy, gen_rat)
        tgt = build_target_text(rec, self.cfg.strategy)

        # No padding here: the collate function pads each batch to its own
        # longest member. Padding every example to source_len=512 wasted ~89%
        # of encoder positions -- real inputs are 11-78 tokens. source_len and
        # target_len now act purely as truncation caps.
        enc = self.tok(
            src, max_length=self.cfg.source_len,
            padding=False, truncation=True, return_tensors="np",
        )
        lab = self.tok(
            tgt, max_length=self.cfg.target_len,
            padding=False, truncation=True, return_tensors="np",
        )
        labels = lab["input_ids"][0].astype(np.int64)
        # -100 masks pad tokens out of the loss (Eq. 6 sums over real tokens).
        labels[labels == self.tok.pad_token_id] = -100
        # An entirely-masked row makes cross-entropy average over zero elements
        # and return NaN, which then contaminates the whole batch. It happens
        # for empty targets -- e.g. the one R-RAD open-end record with a blank
        # solution under First-Stage_Reasoning. Keep one real token.
        if not (labels != -100).any():
            labels[0] = self.tok.eos_token_id

        return {
            "input_ids": enc["input_ids"][0].astype(np.int64),
            "attention_mask": enc["attention_mask"][0].astype(np.int64),
            "labels": labels,
            "ans_mask": answer_segment_mask(self.tok, rec, self.cfg.strategy, labels),
            "image_ids": self.image_feature(self.records[self._image_src[i]]),
            "index": i,
        }

    def check_answer_mask(self, n: int = 3) -> bool:
        """Decode the masked answer segment of a few items; True if all match."""
        ok = True
        for i in range(min(n, len(self))):
            item, rec = self[i], self.records[i]
            ids = item["labels"][item["ans_mask"].astype(bool)]
            seg = self.tok.decode(ids, skip_special_tokens=True).strip()
            want = f"{MARK_ANS} {rec['answer']}"
            if self.cfg.strategy in ("First-Stage_Reasoning",):
                want = ""
            hit = bn_normalize(seg) == bn_normalize(want)
            ok &= hit
            print(f"  [ans_mask] {'OK ' if hit else 'BAD'} {seg!r}  (want {want!r})")
        return ok


def collate_factory(pad_token_id: int = 0):
    """
    Pad each batch to its own longest member (dynamic padding).

    Attention masks and the -100 label mask make this exactly equivalent to
    fixed-length padding, at a fraction of the compute.
    """
    import torch

    def _pad(rows: List[np.ndarray], fill: int) -> "torch.Tensor":
        width = max(len(r) for r in rows)
        out = np.full((len(rows), width), fill, dtype=np.int64)
        for i, r in enumerate(rows):
            out[i, :len(r)] = r
        return torch.from_numpy(out)

    def collate(batch: List[dict]) -> dict:
        return {
            "input_ids": _pad([b["input_ids"] for b in batch], pad_token_id),
            "attention_mask": _pad([b["attention_mask"] for b in batch], 0),
            # -100 keeps padded target positions out of the loss (Eq. 6).
            "labels": _pad([b["labels"] for b in batch], -100),
            "ans_mask": _pad([b["ans_mask"] for b in batch], 0),
            "image_ids": torch.from_numpy(np.stack([b["image_ids"] for b in batch])),
            "index": torch.tensor([b["index"] for b in batch]),
        }

    return collate


# --------------------------------------------------------------------------- #
# 7. Model: BanglaT5/mT5 + cross-attention + gated fusion (paper Sec. 3.2)
# --------------------------------------------------------------------------- #

def _torch_bits():
    import torch
    import torch.nn as nn
    from transformers import T5ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    return torch, nn, T5ForConditionalGeneration, BaseModelOutput


_torch, _nn, _T5Base, _BaseModelOutput = (None, None, None, None)
try:  # pragma: no cover - import-time convenience
    _torch, _nn, _T5Base, _BaseModelOutput = _torch_bits()
except Exception:  # torch absent (e.g. docs build)
    pass


if _T5Base is not None:

    class MedThinkMultimodalGeneration(_T5Base):  # type: ignore[misc,valid-type]
        """
        T5 conditional generation with MedThink's visual pathway bolted on.

        Text encoder output F_T is fused with DETR features F_I via
        cross-attention (Eq. 2) and a learned gate (Eq. 3-4) before the decoder
        ever sees it, so the decoder is unchanged and generation is unaffected.
        """

        def __init__(self, config, patch_dim: int = 256, patch_num: int = 100,
                     gate_bias_init: float = -3.0):
            super().__init__(config)
            d = config.d_model
            self.patch_dim = patch_dim
            self.patch_num = patch_num
            self.gate_bias_init = gate_bias_init
            self.image_dense = _nn.Linear(patch_dim, d)
            self.mha_layer = _nn.MultiheadAttention(
                embed_dim=d, num_heads=1, kdim=d, vdim=d, batch_first=True
            )
            self.gate_dense = _nn.Linear(2 * d, d)
            self.sigmoid = _nn.Sigmoid()
            self._init_visual()

        def _init_visual(self):
            """
            Initialise the visual pathway so it starts *closed*.

            With a naive init the gate sits at sigmoid(0) = 0.5, so half the
            pretrained encoder output is replaced by a randomly-projected image
            signal at step 0. That wrecks the pretrained representation before
            the model has learned anything, and on a T5 v1.1 backbone
            (gated-gelu -- both mT5 and BanglaT5) it diverges to NaN within a
            few optimiser steps.

            Instead the gate bias starts at -3 by default (`gate_bias_init`),
            so lambda ~= 0.047: the model
            begins as essentially pure text BanglaT5 and *learns* to open the
            gate to vision. The weight gets a small non-zero init rather than
            zeros -- with exactly zero weights every output channel would
            receive an identical gradient and the gate could never become
            channel-dependent.
            """
            # `from_pretrained` may build the model on the meta device
            # (low_cpu_mem_usage, the modern default). Initialising a meta
            # tensor is a silent no-op, so bail out here and let `load_model`
            # re-run this once the parameters are real.
            mods = [getattr(self, n, None)
                    for n in ("image_dense", "gate_dense", "mha_layer")]
            if any(m is None for m in mods):
                return                      # called before __init__ finished
            if any(p.is_meta for m in mods for p in m.parameters()):
                return                      # still on the meta device

            _nn.init.xavier_uniform_(self.image_dense.weight)
            _nn.init.zeros_(self.image_dense.bias)

            _nn.init.xavier_uniform_(self.gate_dense.weight)
            with _torch.no_grad():
                self.gate_dense.weight.mul_(0.01)   # keep lambda near the bias
            _nn.init.constant_(self.gate_dense.bias,
                               getattr(self, "gate_bias_init", -3.0))

            # nn.MultiheadAttention owns in_proj_weight / out_proj, which are
            # likewise absent from any T5 checkpoint.
            self.mha_layer._reset_parameters()
            # _reset_parameters() initialises in_proj_weight and the biases but
            # NOT out_proj.weight, which can therefore survive meta-device
            # loading as uninitialised memory -- finite by luck on one run and
            # enormous on the next. Initialise it explicitly.
            _nn.init.xavier_uniform_(self.mha_layer.out_proj.weight)
            if self.mha_layer.out_proj.bias is not None:
                _nn.init.zeros_(self.mha_layer.out_proj.bias)

        def _init_weights(self, module):
            """
            Extend T5's initialiser to cover the visual pathway.

            T5PreTrainedModel._init_weights only knows T5's own module types and
            silently ignores anything else, so without this the visual layers
            can survive `from_pretrained` as uninitialised memory -- which shows
            up as +-inf in image_dense and NaN everywhere downstream.
            """
            super()._init_weights(module)
            # This runs during super().__init__() too, before the visual layers
            # are assigned, so every lookup must tolerate their absence.
            visual = [getattr(self, n, None)
                      for n in ("image_dense", "gate_dense", "mha_layer")]
            if any(module is m for m in visual if m is not None):
                self._init_visual()

        def fuse(self, hidden_states, image_ids):
            """Eq. 2-4."""
            img = self.image_dense(image_ids.to(hidden_states.dtype))   # (B,100,d)
            att, _ = self.mha_layer(hidden_states, img, img)            # H_attn
            merge = _torch.cat([hidden_states, att], dim=-1)
            gate = self.sigmoid(self.gate_dense(merge))                 # lambda
            return (1 - gate) * hidden_states + gate * att              # F_fuse

        def encode_with_image(self, input_ids, attention_mask, image_ids):
            """Run the text encoder and fuse vision in. Returns encoder outputs."""
            enc = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            fused = self.fuse(enc.last_hidden_state, image_ids)
            return _BaseModelOutput(last_hidden_state=fused)

        def forward(self, input_ids=None, attention_mask=None, image_ids=None,
                    encoder_outputs=None, labels=None, **kwargs):
            if encoder_outputs is None:
                encoder_outputs = self.encode_with_image(input_ids, attention_mask, image_ids)
            return super().forward(
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                labels=labels,
                **kwargs,
            )

        @_torch.no_grad()
        def generate_with_image(self, input_ids, attention_mask, image_ids, **gen_kwargs):
            """
            Generate with vision fused in.

            The fused encoder output is computed once and handed to `generate`
            as `encoder_outputs`, so HF skips re-encoding. This avoids patching
            `prepare_inputs_for_generation`, which is the part of the original
            MM-CoT/MedThink code most likely to break across transformers
            versions.
            """
            enc = self.encode_with_image(input_ids, attention_mask, image_ids)
            return self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=enc,
                **gen_kwargs,
            )

else:  # pragma: no cover
    class MedThinkMultimodalGeneration:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ImportError("PyTorch/transformers required")


# Backwards-compatible alias (the class is backbone-agnostic).
BanglaT5MultimodalGeneration = MedThinkMultimodalGeneration


def load_model(cfg: Config, tokenizer=None):
    """Instantiate the backbone with the visual pathway and resize embeddings."""
    import torch

    model = MedThinkMultimodalGeneration.from_pretrained(
        cfg.model_name, patch_dim=cfg.patch_dim, patch_num=cfg.patch_num,
        gate_bias_init=cfg.gate_bias_init,
    )

    # T5 v1.1 checkpoints (BanglaT5, mT5) ship an lm_head separate from the
    # input embeddings, but transformers 5 reports tie_word_embeddings=True for
    # BanglaT5 and re-ties them on resize_token_embeddings(): the output layer
    # silently becomes the input embedding table (loss ~1500 instead of ~12,
    # nothing learns). Trust the checkpoint, not the config: if the loaded
    # weights differ they are untied, and must stay so. UnifiedQA (T5 v1.0) is
    # genuinely tied and passes through unchanged.
    untied = (model.lm_head.weight.data_ptr() != model.shared.weight.data_ptr()
              and not torch.equal(model.lm_head.weight, model.shared.weight))
    if untied:
        model.config.tie_word_embeddings = False
    if tokenizer is not None and len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    if untied and model.lm_head.weight.data_ptr() == model.shared.weight.data_ptr():
        raise RuntimeError(
            "lm_head was tied to the input embeddings after loading "
            f"{cfg.model_name}, whose checkpoint keeps them separate. Training "
            "would start from a broken output layer. Check the transformers version."
        )

    # The visual pathway is never present in a T5 checkpoint, so re-initialising
    # it here is always correct -- and necessary, because meta-device loading
    # can leave these tensors as uninitialised memory. Cheap insurance against
    # a silent +-inf that only surfaces as a NaN loss hours into training.
    model._init_visual()

    visual = {n: p for n, p in model.named_parameters()
              if n.startswith(("image_dense", "mha_layer", "gate_dense"))}
    bad = [n for n, p in visual.items() if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(
            f"Non-finite visual parameters after initialisation: {bad}. "
            "This usually means meta-device loading bypassed _init_visual."
        )
    print(f"  visual pathway initialised: {len(visual)} tensors, all finite "
          f"(gate bias {model.gate_dense.bias.mean().item():.2f} "
          f"-> lambda ~ {torch.sigmoid(model.gate_dense.bias).mean().item():.3f})")
    return model


# --------------------------------------------------------------------------- #
# 8. Misc
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def count_parameters(model) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    visual = sum(
        p.numel()
        for n, p in model.named_parameters()
        if n.startswith(("image_dense", "mha_layer", "gate_dense"))
    )
    return {
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
        "visual_head_M": visual / 1e6,
    }


# --------------------------------------------------------------------------- #
# 9. Training / inference engine
# --------------------------------------------------------------------------- #
# Shared by notebooks 02, 03 and 04 so the loop exists in exactly one place.
# Built for Kaggle: every run checkpoints each epoch and honours a wall-clock
# budget, so a session cut off at the 12h limit resumes instead of restarting.

def _build_optimizer(model, cfg: "Config", lr: float):
    import torch
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (n.endswith("bias") or "layer_norm" in n.lower()) else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def _lr_lambda_factory(total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 1.0 - progress)

    return fn


def segment_loss(logits, labels, ans_mask, alpha: Optional[float]):
    """
    Per-segment NLL. Returns (loss, mean answer NLL, mean rationale NLL).

    alpha=None returns the paper's token-mean loss (identical to HF's
    `out.loss`) and only *measures* the two segments -- which is how the
    answer-dilution figure is produced for vanilla runs.

    Otherwise each segment is length-normalised per example and mixed as
    alpha * L_ans + (1 - alpha) * L_rat. An example missing one segment
    (without_R has no rationale) falls back to its plain token mean, so the
    loss scale does not silently shrink by alpha.
    """
    import torch
    import torch.nn.functional as F

    valid = labels.ne(-100)
    tok = F.cross_entropy(logits.float().transpose(1, 2), labels,
                          ignore_index=-100, reduction="none")          # (B, T)
    is_ans = ans_mask.bool() & valid
    is_rat = valid & ~is_ans
    n_ans, n_rat, n_all = is_ans.sum(1), is_rat.sum(1), valid.sum(1)
    l_ans = (tok * is_ans).sum(1) / n_ans.clamp(min=1)
    l_rat = (tok * is_rat).sum(1) / n_rat.clamp(min=1)

    has_ans, has_rat = n_ans > 0, n_rat > 0
    seg_ans = l_ans[has_ans].mean() if has_ans.any() else torch.zeros((), device=tok.device)
    seg_rat = l_rat[has_rat].mean() if has_rat.any() else torch.zeros((), device=tok.device)

    if alpha is None:
        loss = (tok * valid).sum() / valid.sum().clamp(min=1)
    else:
        mixed = alpha * l_ans + (1 - alpha) * l_rat
        plain = (tok * valid).sum(1) / n_all.clamp(min=1)
        loss = torch.where(has_ans & has_rat, mixed, plain).mean()
    return loss, seg_ans.detach(), seg_rat.detach()


def _is_visual_param(name: str) -> bool:
    return name.startswith(("image_dense", "mha_layer", "gate_dense"))


def train_model(
    model,
    train_ds: "MedThinkDataset",
    cfg: "Config",
    lr: Optional[float] = None,
    epochs: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    time_budget_h: float = 10.5,
    log_every: int = 50,
    resume: bool = True,
    save_optimizer: bool = True,
    resume_from: Optional[str] = None,
    val_ds: Optional["MedThinkDataset"] = None,
):
    """
    Fine-tune with the paper's objective (Eq. 6), or the segment-balanced
    variant when `cfg.ans_weight` is set.

    With `val_ds`, the model is scored on it every `cfg.eval_every` epochs
    (see `evaluate_val`), the best weights are kept, and training stops after
    `cfg.resolved_patience()` evaluations without improvement. The returned
    model carries the BEST weights, not the last.

    Returns (model, info). `info` records per-epoch loss, elapsed time and
    whether the wall-clock budget cut the run short, so a partially trained
    result is never reported as if it were complete. Early stopping is a
    completed run (`early_stopped`), not a truncated one (`stopped_early`).
    """
    import time
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    lr = cfg.lr if lr is None else lr
    epochs = cfg.resolved_epochs() if epochs is None else epochs

    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_factory(train_ds.tok.pad_token_id),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    steps_per_epoch = max(1, math.ceil(len(loader) / cfg.grad_accum))
    total_steps = steps_per_epoch * epochs

    optimizer = _build_optimizer(model, cfg, lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda_factory(total_steps, cfg.warmup_ratio)
    )

    start_epoch, history = 0, []
    patience = cfg.resolved_patience() if val_ds is not None else 0
    es_state = {"best": None, "best_epoch": None, "bad": 0,
                "val_history": [], "early_stopped": False}
    # Read the checkpoint from wherever it lives (possibly a read-only
    # /kaggle/input mount); write only ever goes to ckpt_path.
    _load_from = resume_from or ckpt_path
    if resume and _load_from and os.path.exists(_load_from):
        state = torch.load(_load_from, map_location=device)
        saved = state.get("config", {})
        # A checkpoint from a quick_run was trained on a 200-example subset for
        # a couple of epochs. Silently resuming a full run from it would carry
        # those weights and epoch counter into the real result. Refuse.
        mismatch = [
            k for k in ("quick_run", "dataset", "subset", "strategy", "model_name",
                        "protocol", "seed", "ans_weight", "gate_bias_init",
                        "vision_warmup_epochs")
            if k in saved and saved[k] != getattr(cfg, k)
        ]
        if mismatch:
            print(f"  [resume] IGNORING checkpoint -- differs in {mismatch}.")
            print(f"           Training from scratch. Delete {ckpt_path} to silence this.")
        else:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"] + 1
            history = state.get("history", [])
            es_state = state.get("early_stopping", es_state)
            if state.get("finished"):
                start_epoch = epochs              # nothing left to train
                print(f"  [resume] run already finished at epoch {state['epoch']+1}"
                      " -- skipping training")
            else:
                print(f"  [resume] restored epoch {state['epoch']} -> continuing at {start_epoch}")

    # Fail in seconds, not hours: probe that the checkpoint directory is really
    # writable before training starts. Kaggle remounts /kaggle/working
    # read-only once its ~20GB quota is exceeded, and each checkpoint is ~3GB.
    if ckpt_path:
        import shutil as _sh
        # A checkpoint may be READ from a read-only /kaggle/input mount, but it
        # can never be WRITTEN there -- relocate rather than lose checkpointing.
        ckpt_path = ensure_writable_ckpt(ckpt_path)
        best_path = ckpt_path[:-3] + "_best.pt"
        where = os.path.dirname(ckpt_path) or "."
        try:
            free_gb = _sh.disk_usage(where).free / 1e9
            need = 3.0 if save_optimizer else 1.0
            print("  [ckpt] %s writable, %.1fGB free (need ~%.0fGB per save)"
                  % (where, free_gb, need))
            if free_gb < need:
                print("  [warn] only %.1fGB free -- checkpointing will likely fail. "
                      "Pass save_optimizer=False to third the checkpoint size." % free_gb)
        except OSError as e:
            print("  [warn] could not stat %s (%s); continuing." % (where, e))
    else:
        best_path = None

    # A resumed run's best weights live next to the checkpoint it was read
    # from, which may be a read-only /kaggle/input mount.
    best_src = None
    for cand in ([(_load_from[:-3] + "_best.pt")] if _load_from else []) + [best_path]:
        if cand and os.path.exists(cand):
            best_src = cand
            break
    best_state = None                     # in-memory copy, used if disk fails

    t0 = time.time()
    stopped_early = False
    nan_batches = nan_steps = 0
    show_segments = cfg.ans_weight is not None or cfg.strategy in ("Explanation", "Reasoning")

    for epoch in range(start_epoch, epochs):
        model.train()
        running, nb = 0.0, 0
        seg_a, seg_r = 0.0, 0.0
        optimizer.zero_grad(set_to_none=True)

        # Vision warm-up: only the visual pathway trains for the first epochs.
        # The optimiser was built with every parameter, so unfreezing later
        # needs no rebuild; frozen tensors simply receive no gradient.
        warm = epoch < cfg.vision_warmup_epochs
        for n_, p_ in model.named_parameters():
            p_.requires_grad_(True if not warm else _is_visual_param(n_))
        if warm:
            print(f"  [warm-up] epoch {epoch+1}: text backbone frozen, visual pathway only")

        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True)
                     for k, v in batch.items() if k != "index"}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                image_ids=batch["image_ids"],
                labels=batch["labels"],
            )
            loss, l_ans, l_rat = segment_loss(out.logits, batch["labels"],
                                              batch["ans_mask"], cfg.ans_weight)

            # A non-finite loss must never be averaged into the epoch mean and
            # reported as if it were training. Drop the batch instead.
            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if nan_batches == 1:
                    print("    [warn] non-finite loss -- batch skipped. "
                          "If this repeats, lower cfg.lr.")
                continue

            (loss / cfg.grad_accum).backward()
            running += loss.item()
            seg_a += l_ans.item()
            seg_r += l_rat.item()
            nb += 1

            if (step + 1) % cfg.grad_accum == 0 or (step + 1) == len(loader):
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                # Clipping does not rescue inf/nan gradients: stepping on them
                # turns every weight into NaN permanently.
                if torch.isfinite(gnorm):
                    optimizer.step()
                else:
                    nan_steps += 1
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if log_every and (step + 1) % (log_every * cfg.grad_accum) == 0:
                print(f"    ep {epoch+1}/{epochs} step {step+1}/{len(loader)} "
                      f"loss {running/max(nb,1):.4f} lr {scheduler.get_last_lr()[0]:.2e}")

        elapsed_h = (time.time() - t0) / 3600
        history.append({"epoch": epoch, "loss": running / max(nb, 1), "elapsed_h": elapsed_h,
                         # Mean per-example NLL of each segment: the evidence
                         # for (or against) answer-loss dilution.
                         "ans_loss": seg_a / max(nb, 1), "rat_loss": seg_r / max(nb, 1)})
        msg = f"  epoch {epoch+1}/{epochs}  loss {history[-1]['loss']:.4f}"
        if show_segments:
            msg += f"  [ans {history[-1]['ans_loss']:.4f} | rat {history[-1]['rat_loss']:.4f}]"
        print(msg + f"  ({elapsed_h:.2f}h)")

        # ---- validation / early stopping --------------------------------
        if val_ds is not None and ((epoch + 1) % max(cfg.eval_every, 1) == 0
                                   or epoch + 1 == epochs):
            score, sname = evaluate_val(model, val_ds, cfg)
            es_state["val_history"].append({"epoch": epoch, sname: score})
            improved = es_state["best"] is None or score > es_state["best"]
            if improved:
                es_state.update(best=score, best_epoch=epoch, bad=0)
                best_state = {k: v.detach().to("cpu", copy=True)
                              for k, v in model.state_dict().items()}
                if best_path:
                    try:
                        tmp = best_path + ".tmp"
                        torch.save(best_state, tmp)
                        os.replace(tmp, best_path)
                        best_src = best_path
                    except OSError as e:
                        print(f"  [warn] cannot write best weights ({e}); kept in memory only.")
            else:
                es_state["bad"] += 1
            print(f"  [val] {sname} {score:.2f}  best {es_state['best']:.2f} "
                  f"@ epoch {es_state['best_epoch']+1}"
                  + ("" if improved else f"  ({es_state['bad']}/{patience or '-'} without gain)"))
            if patience and es_state["bad"] >= patience and epoch + 1 >= cfg.resolved_min_epochs():
                es_state["early_stopped"] = True

        if ckpt_path:
            payload = {
                "model": model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "history": history,
                "config": asdict(cfg),
                "early_stopping": es_state,
                "finished": es_state["early_stopped"] or epoch + 1 == epochs,
            }
            # The AdamW moments are two thirds of a ~3GB checkpoint. Dropping
            # them roughly thirds the file; resume then restarts the optimiser
            # state, which costs a little convergence but survives a tight disk.
            if save_optimizer:
                payload["optimizer"] = optimizer.state_dict()
            try:
                # Write-then-rename: a failure mid-write must not destroy the
                # checkpoint that is already on disk.
                tmp = ckpt_path + ".tmp"
                torch.save(payload, tmp)
                os.replace(tmp, ckpt_path)
            except OSError as e:
                # Losing checkpointing must NOT kill hours of training. Warn
                # once, stop trying, and let the run finish -- the model is
                # still in memory and its results are still computed below.
                print(f"\n  [warn] cannot write checkpoint ({e.__class__.__name__}: {e}).")
                print("         Training CONTINUES but this run can no longer resume.")
                print("         Usually /kaggle/working is full: each checkpoint is ~3GB.")
                print("         Free space, or pass save_optimizer=False to third the size.")
                for stale in (ckpt_path + ".tmp",):
                    try:
                        os.remove(stale)
                    except OSError:
                        pass
                ckpt_path = None

        if es_state["early_stopped"]:
            print(f"  [early stop] no val gain for {patience} evaluations -- "
                  f"stopping at epoch {epoch+1}/{epochs}")
            break

        if elapsed_h > time_budget_h:
            print(f"  [budget] {elapsed_h:.2f}h exceeds {time_budget_h}h budget -- stopping "
                  f"at epoch {epoch+1}/{epochs}. Re-run the notebook to resume.")
            stopped_early = True
            break

    if nan_batches or nan_steps:
        print(f"\n  [warn] {nan_batches} non-finite batches, {nan_steps} skipped steps. "
              "Treat these results as unreliable and lower cfg.lr.")

    # Leave every parameter trainable for whoever uses the model next.
    for p_ in model.parameters():
        p_.requires_grad_(True)

    # Hand back the best-on-val weights, not the last epoch's -- unless the
    # run was cut by the time budget, in which case it resumes from the last.
    if es_state["best_epoch"] is not None and not stopped_early:
        if best_state is None and best_src:
            best_state = torch.load(best_src, map_location="cpu")
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"  [best] restored weights from epoch {es_state['best_epoch']+1} "
                  f"(val {es_state['best']:.2f})")

    return model, {
        "history": history,
        "epochs_completed": (history[-1]["epoch"] + 1) if history else 0,
        "epochs_target": epochs,
        "stopped_early": stopped_early,
        "nan_batches": nan_batches,
        "nan_steps": nan_steps,
        "diverged": bool(history) and not math.isfinite(history[-1]["loss"]),
        "early_stopped": es_state["early_stopped"],
        "best_epoch": None if es_state["best_epoch"] is None else es_state["best_epoch"] + 1,
        "best_val": es_state["best"],
        "val_history": es_state["val_history"],
    }


def diagnose_forward(model, ds: "MedThinkDataset", cfg: "Config", n: int = 4) -> bool:
    """
    Run one forward pass and report where non-finite values first appear.

    Pinpoints a NaN instead of guessing: if the encoder output is already bad
    the problem is the backbone or the input; if only `fused` is bad it is the
    visual pathway; if only the loss is bad it is the labels.
    Returns True when everything is finite.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    batch = collate_factory(ds.tok.pad_token_id)([ds[i] for i in range(min(n, len(ds)))])
    batch = {k: v.to(device) for k, v in batch.items() if k != "index"}

    ok = True
    with torch.no_grad():
        stages = {}
        stages["image_ids"] = batch["image_ids"]
        enc = model.encoder(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"], return_dict=True)
        stages["encoder_out"] = enc.last_hidden_state
        stages["image_dense"] = model.image_dense(batch["image_ids"])
        att, _ = model.mha_layer(enc.last_hidden_state,
                                 stages["image_dense"], stages["image_dense"])
        stages["cross_attn"] = att
        merge = torch.cat([enc.last_hidden_state, att], dim=-1)
        stages["gate"] = model.sigmoid(model.gate_dense(merge))
        stages["fused"] = model.fuse(enc.last_hidden_state, batch["image_ids"])

        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    image_ids=batch["image_ids"], labels=batch["labels"])
        stages["logits"] = out.logits
        stages["loss"] = out.loss

        print(f"{'stage':14s}{'finite':>8s}{'mean':>12s}{'std':>12s}{'absmax':>12s}")
        print("-" * 58)
        for name, t in stages.items():
            fin = bool(torch.isfinite(t).all())
            ok &= fin
            f = t[torch.isfinite(t)].float()
            print(f"{name:14s}{('yes' if fin else 'NO'):>8s}"
                  f"{(f.mean().item() if f.numel() else float('nan')):>12.4f}"
                  f"{(f.std().item() if f.numel() > 1 else 0.0):>12.4f}"
                  f"{(f.abs().max().item() if f.numel() else float('nan')):>12.2f}")

    lam = stages["gate"].float().mean().item()
    init = 1 / (1 + math.exp(-getattr(model, "gate_bias_init", -3.0)))
    print(f"\nmean gate lambda = {lam:.4f}  (should start near {init:.3f}: the visual")
    print("pathway opens gradually rather than overwriting the pretrained encoder)")
    print("\nALL FINITE" if ok else "\nNON-FINITE VALUES PRESENT -- see the first 'NO' above")
    return ok


def generate_predictions(
    model,
    ds: "MedThinkDataset",
    cfg: "Config",
    batch_size: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> List[str]:
    """Decode over a dataset, returned in original record order."""
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    loader = DataLoader(
        ds, batch_size=batch_size or max(cfg.batch_size * 2, 8), shuffle=False,
        collate_fn=collate_factory(ds.tok.pad_token_id),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    preds: List[Tuple[int, str]] = []

    with torch.no_grad():
        for batch in loader:
            idx = batch["index"].tolist()
            out = model.generate_with_image(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                image_ids=batch["image_ids"].to(device),
                max_length=max_new_tokens or cfg.target_len,
                num_beams=cfg.num_beams,
                early_stopping=cfg.num_beams > 1,
            )
            preds.extend(zip(idx, ds.tok.batch_decode(out, skip_special_tokens=True)))

    preds.sort(key=lambda x: x[0])
    return [t for _, t in preds]


ANSWER_FIRST = ("without_R", "Explanation", "Second-Stage_Reasoning")


def _choice_label_ids(tokenizer, option: str, strategy: str) -> List[int]:
    """
    Target prefix whose likelihood scores one option.

    Each candidate is scored up to the same boundary -- </s> for answer-only
    targets, the rationale marker for Explanation -- so options of different
    token lengths are compared as complete prefixes, not truncated ones.
    """
    if strategy == "Explanation":
        return tokenizer(f"{MARK_ANS} {option} {MARK_RAT}",
                         add_special_tokens=False)["input_ids"]
    return tokenizer(f"{MARK_ANS} {option}")["input_ids"]          # + </s>


def score_choices(model, ds: "MedThinkDataset", cfg: "Config",
                  batch_size: int = 16) -> List[str]:
    """
    Closed-end prediction by likelihood: pick the option whose answer prefix
    the model scores highest, instead of snapping free-form generation.

    Never unresolved, deterministic, and ~an order of magnitude faster than
    generating a rationale. Valid only when the answer precedes the rationale
    (ANSWER_FIRST); for Reasoning the answer is conditioned on a generated
    rationale and must be decoded.
    """
    import torch

    assert cfg.strategy in ANSWER_FIRST, f"choice scoring needs an answer-first strategy, got {cfg.strategy}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    pad = ds.tok.pad_token_id
    picks: List[str] = []

    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            idx = list(range(start, min(start + batch_size, len(ds))))
            rows, labs, owner = [], [], []
            for i in idx:
                item, opts = ds[i], ds.records[i]["options"]
                assert opts, "choice scoring needs options (closed-end only)"
                for o in opts:
                    rows.append(item)
                    labs.append(np.asarray(_choice_label_ids(ds.tok, o, cfg.strategy), dtype=np.int64))
                    owner.append((i, o))

            batch = collate_factory(pad)([dict(r, labels=l, ans_mask=np.zeros_like(l))
                                          for r, l in zip(rows, labs)])
            labels = batch["labels"].to(device)
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        image_ids=batch["image_ids"].to(device),
                        labels=labels)
            lp = torch.log_softmax(out.logits.float(), dim=-1)
            tok_lp = lp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            seq_lp = (tok_lp * labels.ne(-100)).sum(1).tolist()

            best: Dict[int, Tuple[float, str]] = {}
            for (i, o), sc in zip(owner, seq_lp):
                if i not in best or sc > best[i][0]:
                    best[i] = (sc, o)
            picks.extend(best[i][1] for i in idx)
    return picks


def evaluate_val(model, ds: "MedThinkDataset", cfg: "Config") -> Tuple[float, str]:
    """
    One model-selection score, higher is better. Returns (score, name).

      closed-end, answer-first : choice-scoring accuracy (no decoding)
      closed-end, Reasoning    : decoded accuracy
      open-end                 : answer token-F1

    Answer-first strategies decode only 32 tokens: the answer is complete
    long before the rationale, and selection never looks at the rationale.
    """
    recs = ds.records
    gold = [r["answer"] for r in recs]
    if cfg.subset == "closed-end" and cfg.strategy in ANSWER_FIRST:
        preds = score_choices(model, ds, cfg)
        return evaluate_closed_end(preds, gold, [r["options"] for r in recs])["Accuracy"], \
            "val/ChoiceAccuracy"

    short = 32 if cfg.strategy in ANSWER_FIRST else None
    gens = generate_predictions(model, ds, cfg, max_new_tokens=short)
    answers = [parse_answer(g, cfg.strategy)[0] for g in gens]
    if cfg.subset == "closed-end":
        return evaluate_closed_end(answers, gold, [r["options"] for r in recs])["Accuracy"], \
            "val/Accuracy"
    return token_f1(answers, gold), "val/ans_TokenF1"


def image_ablation(model, ds: "MedThinkDataset", cfg: "Config",
                   modes: Sequence[str] = ("real", "zero", "shuffle")) -> Dict[str, float]:
    """
    Score the same model with its images removed or swapped.

    If accuracy barely moves when the image is zeroed or replaced by another
    patient's, the model answers from the question alone, and no claim about
    visual reasoning is supported. Uses the `evaluate_val` score so the
    numbers are directly comparable across modes.
    """
    out: Dict[str, float] = {}
    try:
        for mode in modes:
            same = ds.set_image_mode(mode)
            score, name = evaluate_val(model, ds, cfg)
            out[f"ablation/{mode}/{name.split('/', 1)[1]}"] = score
            if mode == "shuffle":
                out["ablation/shuffle_same_image_frac"] = 100.0 * same
    finally:
        ds.set_image_mode("real")
    return out


def measure_gate(model, ds: "MedThinkDataset", cfg: "Config", n: int = 64) -> Dict[str, float]:
    """
    How far has the visual gate opened?

    lambda starts at ~0.047 by construction. If it is still ~0.047 after
    training, the model has learned to ignore the images and is a text-only
    model wearing a multimodal architecture -- which would invalidate any
    claim about rationale-guided *multimodal* reasoning. Report this.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batch = collate_factory(ds.tok.pad_token_id)([ds[i] for i in range(min(n, len(ds)))])

    with torch.no_grad():
        enc = model.encoder(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            return_dict=True,
        )
        img = model.image_dense(batch["image_ids"].to(device).to(enc.last_hidden_state.dtype))
        att, _ = model.mha_layer(enc.last_hidden_state, img, img)
        gate = model.sigmoid(model.gate_dense(
            torch.cat([enc.last_hidden_state, att], dim=-1)))
        # Mask out padding positions, which carry no signal.
        mask = batch["attention_mask"].to(device).unsqueeze(-1).bool()
        real = gate.masked_select(mask)

    return {
        "gate/lambda_mean": real.mean().item(),
        "gate/lambda_max": real.max().item(),
        "gate/shift_from_init": real.mean().item()
                                - 1 / (1 + math.exp(-getattr(model, "gate_bias_init", -3.0))),
    }


def majority_baseline(records: List[dict]) -> float:
    """
    Accuracy of always predicting the most common answer.

    The number every closed-end result must beat. A yes/no test set is rarely
    balanced, so raw accuracy alone cannot distinguish a model that learned
    the task from one that learned the class prior.
    """
    if not records:
        return 0.0
    counts = Counter(r["answer"] for r in records)
    return 100.0 * max(counts.values()) / len(records)


def ensure_writable_ckpt(ckpt_path: str,
                         fallbacks: Sequence[str] = ("/kaggle/working", ".")) -> str:
    """
    Guarantee checkpoints are written somewhere writable.

    Anything attached as a Kaggle Dataset or Model is mounted read-only under
    /kaggle/input. Pointing `ckpt_path` there -- easy to do when wiring up a
    resume from a previously uploaded checkpoint -- makes every save fail. The
    file to *read* may live on a read-only mount; the file to *write* never
    can. Relocate instead of losing checkpointing for the whole run.
    """
    def _writable(d: str) -> bool:
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".mt_probe")
            with open(probe, "wb") as fh:
                fh.write(b"0")
            os.remove(probe)
            return True
        except OSError:
            return False

    d = os.path.dirname(ckpt_path) or "."
    if _writable(d):
        return ckpt_path

    for alt in fallbacks:
        if _writable(alt):
            new = os.path.join(alt, os.path.basename(ckpt_path))
            print(f"  [ckpt] {d} is read-only -- writing checkpoints to {alt} instead")
            return new
    print(f"  [warn] no writable directory found for checkpoints (tried {d}, {list(fallbacks)})")
    return ckpt_path


def find_checkpoint(ckpt_path: str, data_root: str = "/kaggle/input") -> Optional[str]:
    """
    Locate a resumable checkpoint WITHOUT copying it.

    Kaggle wipes /kaggle/working between sessions, so a multi-session run must
    read its checkpoint from the previous session's attached output. The
    obvious approach -- copy it into /kaggle/working -- doubles a ~3GB file and
    was enough to exhaust the working volume, which Kaggle then remounts
    read-only, killing the run at its first save. So: read from wherever it is,
    write only to the working directory.

    Returns a path to load from, or None.
    """
    if os.path.exists(ckpt_path):
        return ckpt_path
    hits = sorted(Path(data_root).glob(f"**/{os.path.basename(ckpt_path)}"))
    if hits:
        print(f"  [resume] found previous checkpoint: {hits[0]}")
        return str(hits[0])
    return None


def record_uid(rec: dict) -> str:
    """
    Globally unique record id: dataset/subset/split/qid.

    `qid` is only unique *within one JSON file*. R-PathVQA numbers its train and
    test splits from 1 independently, so 3257 closed-end qids (and 3292
    open-end) appear in both. Keying the two-stage rationale cache by qid alone
    silently overwrites ~35% of the training rationales with test-derived ones
    -- both a correctness bug and train/test contamination. R-RAD and R-SLAKE
    happen not to collide, which is precisely why this hides until PathVQA.
    """
    return "/".join(str(rec.get(k, "")) for k in ("dataset", "subset", "split", "qid"))


def degenerate_rate(answer_preds: Sequence[str]) -> Dict[str, float]:
    """
    How often the model produced no actual answer.

    A model that emits only the "উত্তর:" marker and stops still scores ~50
    ROUGE-1 against a short `without_R` target, because the marker is half the
    reference. That looks like a result and is not one. This measures the
    failure directly: what fraction of answer spans are empty or punctuation.
    """
    n = max(len(answer_preds), 1)
    empty = sum(1 for a in answer_preds if not strip_punct(bn_normalize(a)))
    lens = [len(bn_tokenize(a)) for a in answer_preds]
    return {
        "EmptyAnswerRate": 100.0 * empty / n,
        "MeanAnswerTokens": sum(lens) / n,
    }


def constant_answer_baseline(train_recs: List[dict],
                             test_recs: List[dict]) -> Tuple[str, Dict[str, float]]:
    """
    Score of always emitting the single most common training answer.

    The open-end analogue of the majority-class baseline. Open-end BLEU/ROUGE
    can look respectable purely from reproducing the answer *style* without
    answering anything, so this is the floor a real result must clear.

    Returns (the constant answer, its metrics).
    """
    if not train_recs or not test_recs:
        return "", {}
    const = Counter(r["answer"] for r in train_recs).most_common(1)[0][0]
    scores = evaluate_open_end([const] * len(test_recs),
                               [r["answer"] for r in test_recs])
    return const, {f"baseline/{k}": v for k, v in scores.items()}


def save_results(cfg: "Config", metrics: Dict[str, float], records: List[dict],
                 predictions: List[str], train_info: dict,
                 out_dir: Optional[str] = None) -> str:
    """Write metrics + per-example predictions for notebook 05 to aggregate."""
    out_dir = out_dir or cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    name = cfg.run_name()

    mpath = os.path.join(out_dir, f"metrics_{name}.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump({"run": name, "config": asdict(cfg), "metrics": metrics,
                   "train": train_info}, fh, ensure_ascii=False, indent=2)

    ppath = os.path.join(out_dir, f"preds_{name}.jsonl")
    with open(ppath, "w", encoding="utf-8") as fh:
        for rec, pred in zip(records, predictions):
            ans, rat = parse_answer(pred, cfg.strategy)
            fh.write(json.dumps({
                "qid": rec["qid"],
                "question": rec["question"],
                "gold_answer": rec["answer"],
                "options": rec["options"],
                "pred_raw": pred,
                "pred_answer": ans,
                "pred_rationale": rat,
                "gold_rationale": rec["solution"],
                "image_key": rec["image_key"],
                "image_organ": rec.get("image_organ", ""),
            }, ensure_ascii=False) + "\n")

    print(f"  wrote {mpath}")
    print(f"  wrote {ppath}")
    return mpath
