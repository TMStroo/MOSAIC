# M6: Label Protocol and Operating-Point Selection

**Status:** Phase 2 complete (statistical baselines). Phase 3 (ML baselines) not started.

**Scope:** how the definition of "an anomaly" and the choice of operating
objective change detector conclusions when evaluation is strictly out-of-time.

**Secondary research question, added during M6:**

> How sensitive are anomaly-detection conclusions to the definition of what
> constitutes an observable anomaly?

This document records the answers so far. It reports what was measured,
including results that are unflattering to the detectors.

---

## 1. The two label protocols

`mosaic.labels.resolve` resolves ground truth once and emits two labelled
frames from a single pass (`resolve_both`).

| | `EVENT_ONLY` | `ALL_FAMILIES` |
|---|---|---|
| scope | anomalies with an observable event-level manifestation | every family, including entity- and window-scoped |
| positives | 163 | 18,098 |
| prevalence | 0.033% | 3.63% |
| train / validation / backtest / forward | 76 / 24 / 42 / **21** | 1,797 / 13,828 / 1,582 / 891 |

**Neither is declared correct.** They answer different questions:

- `EVENT_ONLY` asks *can a detector find an anomaly that a human could
  recognise on the event itself?* It is semantically clean, and it has almost
  no forward statistical power.
- `ALL_FAMILIES` asks *can a detector tell an anomalous period from an ordinary
  one?* Its support is 55x larger, but a positive partly means "this row
  belonged to an entity that was ever flagged", which is not the same claim.

### 1.1 Observability is a property of the world, not a join defect

Only 188 of 709 event-kind ground-truth labels are observable at all: the
generator emits roughly 29% of latent events to any source. `entity_resolution`
and `missingness` have no per-event manifestation and resolve only through
entity/window scope, which is why they exist only in `ALL_FAMILIES`.

This is recorded in every artifact under `observability`. It is never absorbed
into a denominator silently.

### 1.2 The `collective` label was wrong until this milestone

The persisted label for `collective` stores `started_at`/`ended_at` as an
envelope over ~45 donor events drawn from ~45 distinct entities. Those donors
are scattered across up to 2,441 hours, so span-based resolution marked
**183,841 rows (≈37% of the dataset)** as anomalous. That is not a detector
result; it is a labelling defect.

Fix: the generator now persists per-event `fam_<family>` flags
(`generator.py::_latent_frame`), and resolution uses those flags. `collective`
is 617 flagged events, 120 observable.

Verified additive: all four source Parquet checksums are byte-identical before
and after regeneration, and every pre-existing `latent_events` column is
byte-identical. Only the truth sidecar gained columns.

---

## 2. E001 — statistical baselines

Five detectors (`mosaic.detection.statistical`), one feature each, threshold
selected on validation only and frozen before backtest and forward.

All five are deterministic: no sampling, no seed dependence. None produces a
calibrated probability, and `predict_proba` raises rather than returning a
score under a probability-shaped name.

### 2.1 The headline result is negative

Under `ALL_FAMILIES`, **four of five detectors select a threshold of exactly
0.0**, flag all 71,674 forward rows, and reach precision 0.0124 — which is
precisely the base rate. Recall is 1.0 and is vacuous at that operating point.

**A threshold of 0.0 that flags every row is a degenerate operating point, not
successful anomaly detection.** Precision equal to prevalence means the
detector's ranking carries no information at the selected cut; the classifier
could have been replaced by a constant.

**No detector was modified to remove this.** It is reported, pinned by
`TestRegressionAllFamiliesZeroThreshold` in
`tests/leakage/test_threshold_leakage.py`, and diagnosed in §4.

### 2.2 `EVENT_ONLY` forward period

| detector | precision [95% CI] | recall [95% CI] | rows flagged |
|---|---|---|---|
| ewma | 0.00034 [0.00021, 0.00058] | 0.667 [0.454, 0.828] | 40,662 |
| robust_z | 0.00040 [0.00020, 0.00087] | 0.333 | 16,644 |
| iqr | 0.00040 [0.00020, 0.00096] | 0.286 | 13,644 |
| changepoint | 0.00022 [0.00010, 0.00053] | 0.238 | 22,285 |
| zscore | 0.0 [0, 0.00142] | 0.0 | 2,693 |

**These numbers must not be used to rank the detectors.** With 21 forward
positives, a single event moves recall by 0.048. The confidence interval on
ewma's recall spans 0.37 points, and every detector's interval overlaps every
other's. The ordering above is not statistically meaningful and is not claimed
to be.

---

## 3. Threshold selection is an explicit experimental variable

`mosaic.detection.thresholds` implements four objectives. F1 maximisation is
the default, not a recommendation:

| objective | parameter | semantics |
|---|---|---|
| `f1_max` | — | maximise F1 over candidate cuts |
| `precision_constrained` | `min_precision` | smallest flag volume meeting a precision floor |
| `recall_constrained` | `min_recall` | smallest flag volume meeting a recall floor |
| `fixed_fpr` | `target_fpr` | largest flag volume under an FPR ceiling |

An unmet constraint yields `feasible=False` with a reason. It is never quietly
relaxed, and `freeze()` raises rather than freezing an infeasible result.

**Validation-only selection is enforced structurally, not by convention.**
`assert_validation_only` raises `LeakageError` for every period except
`validation`, and `select_threshold` calls it before any candidate is
evaluated. `tests/leakage/test_threshold_leakage.py` includes the adversarial
cases: selecting on `forward`, `backtest`, `train`, `unassigned` and the empty
string all raise.

---

## 4. E006 — is the objective responsible for the degeneration?

E006 holds the events, labels, split plan, fitted state, features, detectors,
detector scores, validation period, metrics and periods **all fixed**, and
varies only the objective. Scores are computed once per (detector, period) and
reused across all four objectives, so any difference is caused by the cut, not
the model.

**The degeneration follows the objective.**

| detector | `f1_max` | `precision_constrained` | `recall_constrained` | `fixed_fpr` |
|---|---|---|---|---|
| zscore | thr 0, 71674 | INFEASIBLE | thr 0, 71674 | 761 |
| robust_z | thr 0, 71674 | INFEASIBLE | thr 0, 71674 | 758 |
| iqr | thr 0, 71674 | INFEASIBLE | thr 0, 71674 | 804 |
| ewma | thr 0, 71674 | INFEASIBLE | thr 0, 71674 | 827 |
| changepoint | 343.3, 44537 | INFEASIBLE | thr 0, 71674 | 2057 |

(`ALL_FAMILIES`, forward rows flagged. `fixed_fpr` at target 0.01.)

Three conclusions, each of which is a result:

1. **F1 maximisation and recall maximisation both degenerate to flagging
   everything.** On `ALL_FAMILIES` a positive is rare enough inside any single
   validation window that F1's optimum is the all-rows cut. Recall ≥ 1.0 forces
   it by construction. This is the objective behaving as defined — which is
   exactly why it must be chosen deliberately and reported.

2. **A precision constraint is infeasible on both protocols, for every
   detector.** Not one validation threshold reaches precision 0.5. This is the
   honest answer: at the base rates present, no detector concentrates
   positives enough for a precision floor to be satisfiable. The guard reports
   that rather than lowering the bar.

3. **An FPR ceiling produces a non-degenerate operating point for every
   detector.** Under `fixed_fpr` at target 0.01, all five flag 758–2,057 rows
   instead of 71,674. Forward precision still lands at 0.008–0.017 — at or
   below the `ALL_FAMILIES` base rate of 0.0124 — so **fixing the degeneracy
   did not produce a useful detector.** Removing the degenerate cut removed a
   false appearance of success, not the underlying failure.

Point 3 is the important one: the degenerate operating point was hiding how
little signal these detectors carry. A constraint that forces a defensible
flag volume exposes that.

### 4.1 Under `EVENT_ONLY`

`fixed_fpr` yields 758–1,940 flagged rows, forward precision 0.0 for four
detectors and 0.00052 for changepoint, on 21 positives. `precision_constrained`
is infeasible here too. `recall_constrained` again degenerates, except for
changepoint (thr 17.64, 68,374 rows, recall 1.0 — still 68k rows).

---

## 5. Conclusions

1. **Label semantics and operating-point choice are both explicit experimental
   variables.** Changing only the label protocol, with every detector, feature,
   split and metric held fixed, inverts not just the numbers but the meaning of
   the operating point. Changing only the objective moves the flag volume by
   two orders of magnitude. Neither can be left implicit.

2. **`EVENT_ONLY` has stronger semantic cleanliness and inadequate forward
   statistical power.** 21 forward positives give a ±0.37 recall interval. It
   is the right primary definition and the wrong basis for a ranking.

3. **`ALL_FAMILIES` has much higher support and produces degenerate operating
   points under F1 threshold selection.** Its 891 forward positives are largely
   entity-membership scope rather than event-level anomaly, and its positives
   are concentrated in one validation window, so validation-tuned thresholds do
   not transfer to forward.

4. **Statistical baselines do not detect these anomalies in any useful sense.**
   Once the operating point is forced to a defensible flag volume, forward
   precision is 0.008–0.017 against a 0.0124 base rate. This is a negative
   result and is the reason Phase 3 exists.

5. **A zero threshold is a finding, not a defect.** It was diagnosed rather
   than patched.

### 5.1 What is deliberately not concluded

- Neither label protocol is called correct.
- The detectors are not ranked on forward recall; the intervals do not support
  it.
- Nothing here is evidence about real-world anomaly detection. The data is
  synthetic (`generator_version` recorded in every artifact) and findings are
  labelled as such.
- `collective` and window-scoped families are *definitions*, not ground truth
  disputes, and `ALL_FAMILIES` numbers are reported as a sensitivity analysis.

---

## 6. Standing constraints

Unchanged and re-verified by `pytest tests/` and `pytest tests/leakage/`:

- entity-resolution negative result (probabilistic P=1.000 / R=0.4918)
- `graph_new_neighbor_ratio` saturation (0.049057 → 8.1e-05)
- leakage protocol and temporal semantics

Not changed to make either protocol look better. `ALL_FAMILIES` is not reported
selectively: its degenerate results are stated with the same prominence as its
usable ones.

---

## 7. Reproducing

```
.venv/Scripts/python.exe scratch/run_e001.py   # statistical baselines, both protocols
.venv/Scripts/python.exe scratch/run_e006.py   # threshold-objective comparison
```

Artifacts: `docs/experiments/E001/` (per detector × protocol),
`docs/experiments/E006/` (`shared.json`, `summary.json`).

Each artifact records experiment ID, detector, label protocol, feature set,
periods, threshold method, threshold, metrics, confidence intervals, runtime
and dataset/feature version.