# Annotating LP-2025

CCPD2019 needs none of this: its filenames carry the four plate vertices and the
character indices, so `eval/ccpd/build_ccpd_conditions.py` reads the annotation
straight off the filename. LP-2025 ships the plate string but no geometry, so the
text quadrilateral has to be detected. This page covers that pipeline.

**You probably do not need to run it.** The conditions for all three LP splits —
2,569 train, 620 val, 3,258 test — are published as a release asset. Run this only
to annotate photographs of your own.

## Stage 1 — detection

DeepSolo++ with the multilingual finetune checkpoint. Other detectors were tried
and rejected: general OCR models return loose rectangles that do not follow text
boundaries; YOLO11s with an oriented-box head cannot express the quadrilateral
that perspective produces; its segmentation variant throws outlier points that
destabilise quadrilateral fitting; SAM3 segments the plate perfectly but has no
text-level focus, and shrinking the plate mask by a fixed ratio does not
generalise across Taiwan's plate layouts.

```bash
cd DeepSolo/DeepSolo++
python demo/vis_and_bbox.py \
    --config-file configs/R_50/mlt19_multihead/finetune.yaml \
    --input '<plates>/*.jpg' --output test_res \
    --opts MODEL.WEIGHTS r50_data3_multilingual_finetune.pth
```

Needs `detectron2` and `adet` built — see `DeepSolo/DeepSolo++/README.md`. Writes
one file per plate: `<recognised text> x1 y1 x2 y2 x3 y3 x4 y4`, one line per
polygon. Filters at `MIN_BOX_AREA = 2000` and keeps Latin (`TARGET_LANG_IDS = {6}`).

## Stage 2 — polygons to conditions

```bash
python train/script/annotate_lp.py \
    --det DeepSolo/DeepSolo++/test_res --images data/lp/test/filtered_plate \
    --out data/lp/test --font font/TWGen7_V1.ttf --report annotate.tsv
```

Three steps:

1. **Polygon selection.** Sort the detections by area, take index `len//2` — the
   upper median.
2. **Fuzzy matching.** Raw recognition is unreliable: `AXS9956` comes back as
   `AXS-99556`. The recognised string is aligned to the ground truth from the
   filename by sliding-window edit distance under a visual-confusion dictionary
   (`O0DQ`, `I1L`, `B8`, `S5`, `Z2`, `G6`, `UV` each count as one character).
   The alignment says which substring of the plate the polygon covers. The target
   text always comes from the ground truth — the recogniser's output is used only
   to align.
3. **Context-preserving mask.** If the polygon covers part of the string, use it
   as the mask directly. If it covers all of it, masking everything would leave
   the model no style context, so the polygon is split into `n` equal segments
   (`n` = number of alphanumerics recognised) and a contiguous run of `k < n` is
   sampled at random.

## How faithful is this?

Stage 2 was reconstructed. The original script lived on an external drive that is
no longer attached, so it was rebuilt from the detector's outputs and the released
annotations — both of which survive for all three splits — and then checked
against them.

Reconstruction run over the 3,918 test plates against the 3,258 released annotations:

| | Rebuilt | Released |
|---|---:|---:|
| Input plates | 3,918 | 3,918 |
| No detection | 614 | 614 |
| Rejected by matching | 45 | 46 |
| **Annotated** | **3,259** | **3,258** |

Agreement on the 3,253 shared plates, split by branch:

| Branch | n | Mean mask IoU | Median | IoU > 0.99 | Same target text |
|---|---:|---:|---:|---:|---:|
| Partial coverage — deterministic | 1,351 | **0.9994** | **1.0000** | **99.9%** | 92.2% |
| Full coverage — random span | 1,902 | 0.3837 | 0.3424 | 4.3% | 4.9% |

Everything deterministic reproduces to the pixel. The full-coverage branch cannot:
the original run drew its span at random and the seed was not recorded, so 4.3% is
simply the rate at which two independent draws coincide. The geometry itself is
sound — holding the span choice fixed and comparing only the shape gives a mean
IoU of 0.9612 and a median of 0.9953 across all 3,258.

Two details were recovered by measurement rather than read off the thesis:

* Polygon selection is the **upper** median, `sorted_by_area[len//2]`. Verified on
  every one of the 3,258 released annotations: the released mask falls inside the
  polygon this picks, 3258/3258.
* The full-coverage split uses the **recognised** alphanumeric count, not the
  ground-truth length. Splitting on the ground truth matches 38.2% of released
  annotations at a 2% tolerance; splitting on the recognised count matches 72.2%.

## Label noise

These labels are weakly supervised and contain unintended full masks from matching
failures, slight horizontal boundary shifts, and occasional oversized polygons.
The thesis reports the training framework tolerates this and converges stably
regardless; the released checkpoint was trained on labels of exactly this kind.
