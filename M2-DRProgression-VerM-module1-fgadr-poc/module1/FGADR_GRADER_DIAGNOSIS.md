# Why the FGADR grader scores below chance on Tianjin

**Symptom.** The FGADR grader (`drgnet_replication_prefinals/classification/fgadr/saves/final_weights.pt`)
gets quadratic kappa 0.81 on FGADR's own test set (notebook 11). On Tianjin it gets accuracy about 0.02,
kappa about 0.00 and within-one-grade 0.06-0.11, at the patient and eye levels, in rounds 4 and 5, at full
resolution and at 128/256 px. On the same images the EyePACS grader gets accuracy 0.69 and kappa 0.59.

## What is *not* the cause (checked)

| Suspect | Finding |
|---|---|
| FGADR labels / off-by-one / class order | `make_fgadr_classification_pkl.py --split release` is verified image-by-image (name, order, grade) against DRG-Net's released `fgadr_data_dict.pkl` (`data/drgnet_fgadr_classification_split.csv`). Grades are 0-4 (counts 101/212/595/647/287), and the grader's own FGADR confusion matrix is diagonal-heavy, so the class order is right. |
| Tianjin mapping | `tianjin_grade_to_icdr` maps 1-5 to 0-4, and grades 6/7 are excluded. The same mapping and rows give the EyePACS grader kappa 0.44-0.59, so the comparison is like for like. |
| Normalisation | `CLASSIFIER_NORMS['fgadr']` is byte-identical to `configs/fgadr_poc.yaml` (`data.mean/std`). |
| Inference preprocessing | DRG-Net's test transform (`dr_classification/data/transforms.py` at `0da1bbe`) is PIL RGB -> `Resize((512,512))` -> `ToTensor` -> `Normalize`, the same as `predict_grade`. The retina crop is the only addition, and the EyePACS grader goes through the same crop. The FGADR grader fails just as badly at 128 px as at full resolution. `diagnose_fgadr_grader.py` part C also re-scores it without the crop. |
| Checkpoint loading | Bare `state_dict`, `strict=True`. A wrong head or key prefix would raise an error, not load silently. |

## Cause: FGADR's grade labels are confounded with the image source

FGADR file names end in `_1`, `_2` or `_3`, which is the image source. Grade by source, all 1842 images
(`python module1/diagnose_fgadr_grader.py --labels-only`):

| source | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `_1` | 101 | 212 | 595 | 63 | 29 |
| `_2` | 0 | 0 | 0 | 178 | 76 |
| `_3` | 0 | 0 | 0 | 406 | 182 |

Every `_2`/`_3` image is grade 3 or 4, and 90% of the grade-3/4 images come from `_2`/`_3`. A rule that
looks **only at the file suffix**, with no pixels at all, scores accuracy 0.60 / **qwk 0.59** on DRG-Net's
FGADR test split. The test split has the same confound (`_2`/`_3`: 157 images, all grade 3/4), so
FGADR's own test score rewards a grader that recognises the source instead of grading lesions. Tianjin
comes from none of these sources, so that shortcut doesn't transfer.

The Tianjin numbers fit a grader that gives nearly every Tianjin image the same high grade:

- kappa is about 0, which means the predictions don't vary with the true grade.
- Accuracy (about 0.02) is roughly the share of Tianjin images with clinical grade 4.
- Within-one (0.04 at baseline, 0.11 at follow-up) is roughly the share with grade 3-4. Severe DR is rarer
  at baseline than at follow-up, which matches baseline being lower.
- In notebook 06, the second grader's "grade went up" rate is 0-10% while the primary grader's is
  20-65%, as you'd expect from a grader that is already at the top grade.

**Verdict: a genuine domain failure, not a code bug.** The FGADR grader is not a usable independent
grader for Tianjin.

## Confirmed on Colab (notebook 12, `outputs/12_grader_resolution_check_colab (1).ipynb`)

**Part B: the grader on FGADR's test split, by image source.**

| | n | accuracy | qwk | predicted 0/1/2/3/4 |
|---|---|---|---|---|
| all | 342 | 0.719 | 0.809 | 32/54/83/115/58 |
| source `_1` | 185 | 0.589 | **0.508** | 32/54/83/8/8 |
| sources `_2`/`_3` | 157 | 0.873 | 0.719 | **0/0/0**/107/50 |

- It never predicts below 3 for a `_2`/`_3` image (157/157).
- Among source-`_1` images it predicts 3 or 4 only 16 times out of 185, and catches only 9 of the 18
  grade-3/4 `_1` images (50%).
- Within a single source the grader is much weaker (qwk 0.51 on `_1`). The headline 0.81 relies on
  telling the sources apart.

The `_1` grade-3/4 group is small (n=18), so the 50% figure is rough. The direction is clear, though.

**Part C: the grader on Tianjin vs. clinical grades (per eye, n=2174).**

```
              predicted 0    1    2     3     4
clinical 0 (1682):      0    0    0   581  1101
clinical 1  (189):      0    0    0    95    94
clinical 2  (251):      0    0    0    76   175
clinical 3   (28):      0    0    0     8    20
clinical 4   (24):      0    0    0     2    22
```

- Accuracy 0.014, qwk 0.002.
- **Every Tianjin eye is graded 3 or 4**, including all 1682 eyes with no DR. Mean softmax is 0.40 for
  grade 3 and 0.60 for grade 4, and under 0.01 for grades 0-2.
- Without the retina crop the result is the same (accuracy 0.015, 857 / 1317 predicted 3 / 4), so the
  crop is not the cause.

**Conclusion: the grader learned to recognise the image source.** To it, Tianjin photos look like FGADR's
`_2`/`_3` sources, which in training were always grade 3-4. This is a genuine domain failure caused by
FGADR's source/grade confound, not a code bug.

## Consequence for notebook 06

The `second` columns from `evaluate_trajectory.py --second-classifier-checkpoint <fgadr>` carry no
information about the generated images. The grader barely responds to the input, so "agrees with
follow-up" and "grade went up" mostly reflect Tianjin's grade distribution. Drop or footnote those
columns. A replacement independent grader has to be trained on data with no source/grade confound,
such as EyePACS, IDRiD, APTOS or DDR grading. FGADR could also be retrained on source `_1` only, but
that leaves just 92 grade-3/4 images.
