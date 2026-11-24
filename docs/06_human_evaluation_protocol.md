# 06 — Human Evaluation and Dataset-Quality Protocol

You have 2+ Bengali-speaking clinicians. Use them for two things: (A) quantifying dataset quality and (B) rating model rationales. Budget about 2–3 working days per annotator.

## A. Dataset (translation) quality — supports C1

Because translation was LLM-generated and then fully human-checked, report:

1. **Post-edit statistics (automatic, no annotator time)**, if the raw LLM output was kept (check the `.checkpoints/` files or your translation logs):
   - % of items edited at all (question / answer / rationale separately)
   - HTER or chrF between the raw LLM output and the final text
   - Mean edit distance per field

   If the raw outputs were not kept, state that, and rely on step 2.
2. **Double audit of 300 items** (100 per dataset, stratified by open/closed), shown side by side with the English source:

| Criterion | Scale |
|---|---|
| Adequacy (meaning preserved) | 1–5 |
| Fluency (natural Bengali) | 1–5 |
| Terminology (medical term correct; transliteration vs translation acceptable) | correct / acceptable / wrong |
| Answer preserved (answer unchanged in meaning) | yes / no |

   Report mean scores, % "wrong terminology", and **inter-annotator agreement** (weighted Cohen's κ for 2 annotators, Krippendorff's α for more).
3. Write a **terminology policy** paragraph explaining when English terms (MRI, DWI, CT) were kept in Latin script, transliterated (সিটি) or translated. Reviewers on Bangla papers always ask about this.

Effort: 300 items × ~1 min ≈ 5 h per annotator.

## B. Rationale evaluation — supports C4/C5

**Sample:** 40 test items per dataset (closed-end, stratified: 20 where both systems answer correctly, 20 where they differ) = 120 items.

**Systems, blinded and shuffled per item:**
1. Gold rationale (sanity anchor: it should score highest)
2. MedThink-BN Explanation (vanilla)
3. AARL-full

Each annotator sees the image, Bengali question and system answer + rationale: 120 × 3 = 360 ratings per annotator, about 1.5 min each, so ≈ 9 h.

| Criterion | Question to annotator | Scale |
|---|---|---|
| Answer correctness | Is the answer clinically correct for this image? | yes / no |
| Medical accuracy | Is the rationale free of medical errors? | 1–5 |
| Visual grounding | Does it describe findings actually visible in *this* image? | 1–5 |
| Consistency | Does the rationale support the given answer? | yes / partly / no |
| Language quality | Fluent, correct Bengali medical register? | 1–5 |

**Analysis:**
- Mean ± 95% bootstrap CI per system and criterion.
- Wilcoxon signed-rank AARL vs vanilla (paired by item), Holm-corrected.
- Krippendorff's α per criterion. α < 0.4 should be reported honestly with a discussion.
- Correlation (Spearman) of chrF/BERTScore with human scores. This tells readers whether automatic Bangla rationale metrics are trustworthy, and it is itself a small contribution.

## Logistics

- Build the sheet as CSV/Google Sheet: `item_id, image_link, question_bn, answer, rationale, criteria…`, with system IDs hidden in a separate key file.
- Write a 1-page guideline with 2 worked examples per criterion and run a 10-item calibration round together before the real annotation.
- Ethics: annotators are co-authors or acknowledged; data is public de-identified images. A statement saying so suffices for most journals. Check whether your institution requires IRB exemption for expert annotation.
- **Start creating the sheets in week 3** so annotation runs in week 4 while the final open-end runs train.
