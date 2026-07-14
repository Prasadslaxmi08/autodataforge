# Deterministic Pipeline — Baseline (Phase 5)

The measured baseline every future Agentic-AI feature must beat. Generated from
`scripts/benchmark_suite.py` over six dataset profiles (286 images, 1007
detections), CPU `builtin` backend, Windows / Python 3.13, single process.

**Reading rule:** everything under **Measured** is an observed number from a real
run. Everything under **Assumption / hypothesis** is engineering judgement not yet
proven by measurement. They are kept strictly separate on purpose.

---

## 1. Current strengths — Measured

- **Deterministic & repeatable.** Same dataset → byte-identical quality metrics
  and detection counts across independent runs (`test_pipeline_deterministic`,
  `test_quality_deterministic`).
- **Robust to bad input.** 0 crashes across all profiles; corrupt files are
  quarantined per-item (`test_corrupt_images_quarantined`), duplicates skipped by
  content + perceptual hash.
- **Export is always validated.** Every COCO/YOLO package round-trips (write →
  re-parse → structural compare) before it is reported complete; held on every
  profile and every resolution.
- **Throughput:** 15.1–38.6 images/sec (CPU, single process). Median profile
  (50 imgs) ≈ 26 img/s. Peak RAM 54–59 MB. Peak VRAM ≈ 0 (CPU backend).
- **Resume without duplication.** Re-running import over an already-ingested
  folder imports 0 (`test_ingest_resume_is_idempotent`).

## 2. Known limitations — Measured

- **The verifier approves 100% of annotations (review 0%, reject 0%) on every
  profile.** Root cause: the builtin detector's "confidence" is a geometric
  bounding-box *fill* heuristic (measured avg 0.97–0.998), not a learned score,
  so it never falls into the review/reject bands. **Consequence: the human-review
  budget and triage story cannot yet demonstrate any reduction — nothing is
  flagged, and nothing is independently validated as correct.** This is the
  single most important baseline finding.
- **No accuracy measured.** With no ground-truth labels in the baseline,
  precision / recall / mAP are unknown. The baseline measures throughput,
  resource use, and self-consistency — *not* correctness.
- **Single class.** The builtin detector labels every object `"object"`; it does
  not classify. Real ontologies need a real detector backend.
- **Small-object risk.** The detector drops blobs below 0.15% of image area; the
  error analysis flagged 105/105 detections as "small" on the `small_objects`
  profile — i.e. this backend is at its resolution floor there (proxy metric).

## 3. Measured bottlenecks

Per-stage share of pipeline time (from the raw `benchmarks/phase1_*.json`):

| Stage | Typical share | Scales with |
|---|---|---|
| **Verification** | 29–55% (rises with detection count) | # annotations |
| **Ingest** | 13–43% (dominant on small sets) | # images (fixed per-image decode+hash+re-encode) |
| **Segmentation** | 6–35% (dominant on dense scenes) | # detections (one mask call per box) |
| Detection | 5–22% | image resolution |
| Export | 2–9% | # images |

Throughput is inversely correlated with annotation density (measured: 38.6 img/s
on `small_objects` density 3.6 → 15.1 img/s on `dense_objects` density 9.95).

## 4. Future optimization opportunities — Assumption / hypothesis

_Not yet measured; listed as candidates for later phases to prove or reject._

- Verification and segmentation both loop per-annotation in Python — batching or
  vectorizing them should raise throughput on dense scenes (hypothesis).
- The connected-components detector uses a Python-set union-find; a vectorized
  labeling (e.g. `scipy.ndimage.label`) would likely cut detection time
  (hypothesis, untested — scipy is not currently a dependency).
- A real GPU detector backend would change both accuracy and the timing profile
  entirely; the current numbers are a CPU-classical floor, not a ceiling.
- Ingest re-encodes every image to PNG to strip EXIF; storing original bytes with
  a metadata-only EXIF strip could reduce ingest cost (hypothesis).

## 5. Deliverable index

| Deliverable | Where |
|---|---|
| Baseline Benchmark Report | `benchmarks/baseline_report.md` (generated) |
| Per-run Performance Report | `benchmarks/reports/report_*.md` (generated each run) |
| Dataset Quality Report | `DatasetQualityReport` in each run; per-profile table in the baseline report |
| Error Analysis Report | `ErrorAnalysis` in each run; failure table in the baseline report |
| Resource Utilization | "Resource Utilization" section of each per-run report |
| Comparison Framework | `vds/comparison.py` + `benchmarks/comparison.md` (baseline registered) |
| Updated Test Results | `tests/` — 52 tests (see §7 of this file's companion answers) |

---

## Answers to the phase's five questions

**1. Is the deterministic pipeline production-ready?**
Partially. As *plumbing* — ingest, storage, export validation,
benchmarking, error handling, determinism — yes: it is reliable and measured. As
an *annotation product*, no: it cannot yet be trusted to produce correct labels,
because accuracy is unmeasured and the verifier rubber-stamps everything. It is a
production-ready **baseline**, not a production annotation service.

**2. Biggest weaknesses (measured).**
(a) 100% approval / 0% review — the verifier discriminates nothing because
confidences are a geometric heuristic. (b) Accuracy is entirely unmeasured (no
ground truth). (c) Single-class, classical-CV detection that won't survive real
imagery. (d) Verification + ingest dominate runtime.

**3. Which weaknesses should Agentic AI solve?**
- **Correctness of labels** → real detector/segmenter backends + the **VLM
  Verifier Agent** giving a genuine, independent verdict (replaces the geometric
  heuristic).
- **Meaningful triage / review reduction** → the **Analyst/curation** stage,
  which only becomes non-trivial once confidences are real.
- **Ontology & class definitions** → the **Planner Agent** (brief → LabelingPlan).

**4. Which weaknesses should stay deterministic?**
Ingest/dedup/EXIF, export + validation, versioning/lineage, the state machine,
benchmarking, and the *scoring math* of triage. These have correct answers and
must never be handed to an LLM. Performance optimization (batching, vectorizing
components) is deterministic engineering, not AI.

**5. KPIs that must improve once Planner + Analyst land.**
Tracked by the comparison framework (`benchmarks/comparison.md`), baseline
values in parentheses:
- **Verifier–human agreement** on a golden set (baseline: *unmeasured* — the
  first thing to establish).
- **Review rate** — should become > 0% and *meaningful* (baseline 0%, vacuous).
- **Approval precision** — approved labels that are actually correct (baseline:
  100% approval, unvalidated).
- **Human-review reduction** — % of samples a human must touch for target quality
  (baseline: undefined; the headline metric the whole product targets).
- Throughput/RAM must not regress unacceptably as agents are added (baseline:
  26 img/s median, 55 MB).
