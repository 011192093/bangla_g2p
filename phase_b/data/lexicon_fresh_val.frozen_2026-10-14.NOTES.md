# lexicon_fresh_val.frozen_2026-10-14.tsv

Frozen copy of `lexicon_fresh_val.tsv` as it stood on 2026-10-14, after:
- 600 words hand-reviewed for register + phoneme correctness
  (see `phase_b/src/fresh_val_draft_reg_processed.tsv` for the review trail)
- 87 words that contained a decomposed য়/ড়/ঢ় (base consonant + separate
  nukta mark, instead of the precomposed single codepoint) replaced
  wholesale via `merge_fresh_val.py` + `fresh_val_replacement87.tsv`,
  since the decomposed spelling meant those words had never actually
  been seen by the rule engine, the neural model, or the lexicon
  (see `normalizer.compose_nukta` for the underlying bug)
- Finalized and sanity-checked via `finalize_fresh_val.py`
  (600 rows, 3 columns, zero malformed rows, zero remaining decomposed
  nukta, zero duplicate words)

SHA256: `94b10e7a04d8efba0ccbf890da1ee933fe6c436beeba66b18fff70cad29c62b2`

This file must not be edited in place. Any future correction goes
through a new reviewed draft + `merge_fresh_val.py`, producing a new
dated frozen copy -- never a silent overwrite of this one.

## Checkpoints evaluated against it

Two genuinely different trained models exist in this repo and must not
be confused:

| Checkpoint | SHA256 | Trained |
|---|---|---|
| `phase_b/checkpoints/best_model.pt` (== `checkpoints_backup/best_model_oct02.pt`) | `9242b4925f19637114f0b0f6b0054dee5a934607886d470b1699a1ba5406dafb` | 2026-10-02, on the lexicon BEFORE this session's cleanup pass |
| `phase_b/checkpoints_corrected/best_model.pt` | `38ba59fc9b690a529ce8077bfd5665c2d89a0471a2bbb84596b109d047cdc72a` | 2026-10-13, on the lexicon AFTER cleanup |

Every evaluation run earlier in this session (`confidence_check.py`,
`hybrid_analysis.py`, the first `compare_with_engine.py` run) used the
OLD checkpoint only -- the corrected one had not yet been scored.
