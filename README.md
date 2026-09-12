# MM-Net — joint sleep staging and respiratory-event detection in subacute ischemic stroke

Code, results and figures for the IEEE Access manuscript. Every file here backs
something the paper reports: if a number, table or figure is in the manuscript,
the script that produced it and the result file it came from are both in this
repository.

The manuscript itself is not included here: it is under review, and this repository
carries the code, results and figures behind it.

---

## What the model does

A clinical polysomnogram records far more than the EEG, and in a stroke cohort the
cardiorespiratory channels carry the sleep-disordered breathing that predicts
recovery. MM-Net reads fourteen neural and cardiorespiratory channels and produces
two clinical outputs from one forward pass: the sleep stage of every 30-second
epoch, and a per-epoch respiratory-event label. A direct cardiorespiratory path
carries breathing effort to the respiratory head, bypassing the staging objective.

**iSLEEPS, 99 patients, 92,560 epochs, ten-fold patient-independent, three seeds:**

| | |
|---|---|
| Staging accuracy | 0.739 ± 0.020 |
| Cohen's κ | 0.634 ± 0.024 |
| Respiratory AUC | 0.782 ± 0.038 |
| Average precision | 0.419 ± 0.105 |
| Trainable parameters | 2,764,774 |

---

## Layout

```
model/          mmnet_core.py, the network and the ten-fold engine
foundation/     the experiments, the reimplemented baselines and the analyses
preprocessing/  EDF and annotation files to the cached feature tensors
figures/        the seven figures and the scripts that draw them
baselines/      the StagingSeqNet baseline, which carries its own dependencies
results/        every result file the paper cites
data/           empty; see "Data" below
```

---

## The model

[`model/mmnet_core.py`](model/mmnet_core.py) holds
the whole thing: `MMFeatureNet`, `train_fold`, `run_10fold`, and the HMM decoder.

Two details are worth knowing before reading it.

**Fusion is concatenation.** `MMFeatureNet` also contains a `CrossFusion` branch
with four-head attention, reachable only via `fusion="cross"`. The published model
calls `run_10fold(fusion="concat")`, which is a single `Linear(192 → 128)` over the
concatenated embeddings. The attention variant was evaluated and matched
concatenation to within noise, so the simpler one is what the paper reports and
what this repository runs.

**The pretrained encoder is fed at its own rate.** The working rate is 100 Hz, but
LaBraM was pretrained at 200 Hz with a fixed 3000-sample input, so
`build_labram_cache.py` resamples to 200 Hz and splits each epoch into two 15-second
sub-windows that are encoded separately and mean-pooled. Feeding 100 Hz directly
would be silent and wrong: every rhythm would present at half its frequency.

---

## Reproducing the paper

```bash
pip install -r requirements.txt          # install torch first, see the file

# 1. preprocessing — EDF + annotations to cached tensors
python preprocessing/build_multimodal.py
python preprocessing/extract_mm_features.py
python foundation/build_labram_cache.py

# 2. the headline model, three seeds by ten folds
python foundation/run_final_model.py

# 3. the experiments the paper reports
python foundation/run_ablation_grid.py
python foundation/run_bypass_ablation.py
python foundation/run_external_validation.py
python foundation/run_learning_curve.py
python foundation/run_permutation_importance.py

# 4. the figures
python figures/regen_final_figures.py
```

Every script writes into `results/revision/runs/final/`, and every
file already there was produced by the script of the matching name.

---

## Where each result comes from

| Result file | What it backs |
|---|---|
| `final_model.json` | the headline figures; Tables 4, 8 and 12 |
| `ablation_table.json`, `fold_level_tests.json` | Table 6, the modality ablation and its Holm-corrected *p* |
| `bypass_ablation.json` | the direct cardiorespiratory path |
| `derived_seed42.json` | Figures 4 and 6; Tables 5, 7 and 9 |
| `per_subject_seed42.json` | the per-patient analyses, including the incomplete-montage comparison |
| `neural_only_per_subject.json` | the control showing the montage gap is not about zero-filling |
| `external_isruc.json`, `external_sleepedf.json` | Table 11, zero-shot transfer |
| `tinysleepnet`, `sleeptransformer`, `utime`, `micro`, `resnetse`, `atbilstm` (`_stroke` / `_healthy`) | Tables 3, 4 and 13 |
| `attnsleep_seeded.json`, `isleeps_lstm.json`, `stagingseqnet_seeded.json`, `zeroshot_seeded.json` | Table 4 |
| `healthy_cv.json` | the 0.856 healthy-sleep control |
| `permutation_importance_final.json`, `permutation_significance.json` | the permutation attribution |
| `baseline_paired_tests.json` | the paired comparisons in Section VI |
| `train_pop_norm.json` | the deployable-normalisation check in the Limitations |

---

## Data

`data/` is empty. iSLEEPS is a published corpus and is not redistributed here.

- **iSLEEPS** — Maiti et al., *Polysomnography Dataset for Sleep Analysis in
  Ischemic Stroke Patients*, Scientific Data 13:421 (2026),
  [doi:10.1038/s41597-026-06747-w](https://doi.org/10.1038/s41597-026-06747-w)
- **Sleep-EDF** and **ISRUC-Sleep** are used for external validation and are both
  public.

One recording pair in iSLEEPS is duplicated. It is dropped throughout, which is
why the cohort is 99 and not 100, and the fold assignment enforces that.

---

## What this repository does not contain

The exploratory work. Roughly 130 configurations were trained over the course of
this project and 49 of them are in the architecture sweep; none of that is here
beyond the result files the paper cites. The same applies to abandoned branches,
superseded figures and earlier manuscript formats. If it is not reported, it is not
here.
