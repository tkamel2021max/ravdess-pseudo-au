# Pseudo-Action-Unit Distillation for Actor-Independent Emotion Recognition on RAVDESS

Code, splits and result tables for the paper
*"Pseudo-Action-Unit Distillation for Actor-Independent Emotion Recognition on RAVDESS"*
(K. Tlili, T. Gouasmi, M. Othmani, H. Ltifi — under review).

A lightweight RGB **student** network learns, from face crops, to regress 16 pseudo-AU intensities
derived from MediaPipe blendshapes (the **teacher**) and to classify the 8 RAVDESS emotions.
Backbones, causal temporal encoders and losses are compared one factor at a time under a strict
actor-independent protocol.

> The pseudo-AU labels come from MediaPipe. Agreement with them measures **distillation fidelity**,
> not AU accuracy in the FACS sense.

## Main results (out-of-fold, 1440 videos, 24 actors, 3 seeds)

The paper is organised around three questions.

**RQ1 — fidelity to the MediaPipe pseudo-AUs on unseen actors.** Base model: frame-level ICC(3,1) 0.549 over 14 valid AUs
(0.738 with a CCC+L1 loss); per-actor ICC 0.35–0.62; within-video CCC 0.29 (the global ICC mostly reflects between-video
differences). AU6 and AU9 are never activated by MediaPipe on RAVDESS.

**RQ2 — does pseudo-AU supervision help emotion recognition?**

| Model | Acc. | Macro-F1 | NLL |
|---|---|---|---|
| Multi-task student (base) | 0.584 | 0.577 | 1.46 |
| Direct emotion CNN (no AU supervision) | 0.569 | 0.558 | 1.55 |
| AU bottleneck: LR on the 16 predicted AUs | 0.530 | 0.525 | 1.36 |
| LR on 16 MediaPipe pseudo-AU statistics (tuned) | 0.558 | 0.549 | 1.26 |
| LR on 52 MediaPipe blendshape statistics (tuned) | 0.594 | 0.587 | 1.13 |

Multi-task vs. direct: +1.4 points, actor-level 95 % CI [−0.7, +3.6], p = 0.19 (not established). Retraining with the epoch
selected on the emotion loss alone gives identical models.

**RQ3 — cost.** Full CPU pipeline (BlazeFace + crop + network): ResNet-18 student 19.1 ms/frame, MobileNetV3-S student
7.0 ms/frame, MediaPipe FaceLandmarker (teacher) 16.3 ms/frame; ViT-B/16 network alone 383 ms/frame on CPU.

All tables: [`results/`](results) and [`results/p1a/`](results/p1a); actor splits: [`results/splits.csv`](results/splits.csv).

## Protocol

* RAVDESS audio-visual speech videos (modality 01), 24 actors, 8 emotions, 1440 videos.
* 4 actor-independent, gender-balanced folds (6 test actors each); inside each training set,
  3 actors are held out for model selection; each test fold is evaluated once.
* 3 seeds per configuration; 95 % video-level bootstrap CIs; paired bootstrap tests resampling
  videos or actors, with Holm correction (`scripts/paired_tests.py`).
* Pseudo-AU calibration (99th percentile per AU) is computed on training actors only.

## Reproduce

**Google Colab (recommended, free T4 GPU):** open
[`notebooks/ravdess_colab_single_cell_EN.ipynb`](notebooks/ravdess_colab_single_cell_EN.ipynb),
copy the repository files into a folder named `ravdess_au_project` at the root of *My Drive*,
select a T4 GPU and run the single cell.
Everything is saved to Drive fold by fold; re-running the cell resumes after a disconnection.

**Local machine (Python 3.10–3.12, NVIDIA GPU):**

```bash
pip install -r requirements.txt
python download_ravdess.py --out data/raw                         # Zenodo record 1188976
python extract_features.py --raw data/raw --out data/features --workers 4
python baselines.py --feat data/features
python train_eval.py --exps all --seeds 0 1 2 --folds 4 --epochs 15   # resumable
python measure_latency.py
python make_report.py                                             # tables, figures, numbers.tex
python scripts/paired_tests.py --results results --feat data/features
```

`train_eval.py` uses batch 16 (8 for ViT-B/16: `--exps bb_vit --batch 8`).

Additional analyses of the revised paper:

```bash
python train_eval.py --exps base_sel_emo --seeds 0 1 2      # RQ2 control: epoch selected on emotion NLL
python scripts/p1a_analyses.py                               # RQ1/RQ2 analyses on saved predictions (CPU)
python scripts/pipeline_latency.py data/raw/Actor_01/01-*.mp4  # full-pipeline CPU latency (MediaPipe env)
python scripts/export_splits.py --out results/splits.csv     # actor identifiers per fold and seed
```

On Colab: `notebooks/p1a_analyses_cpu.ipynb` (CPU) and `notebooks/p1b_gpu_controlled_latency.ipynb` (GPU).

| File | Role |
|---|---|
| `extract_features.py` | MediaPipe FaceLandmarker: face crops, 52 blendshapes, 16 raw pseudo-AUs per frame |
| `au_common.py` | AU mapping, folds, calibration, metrics (ICC, CCC, F1), bootstrap |
| `baselines.py` | class prior and logistic regressions on MediaPipe statistics |
| `train_eval.py` | student network, losses (ordinal / L1 / CCC+L1), actor-independent training |
| `measure_latency.py` | network-only latency on CPU and GPU |
| `make_report.py` | LaTeX tables, figures and macros for the paper |
| `scripts/paired_tests.py` | paired video- and actor-level bootstrap tests |
| `scripts/plot_per_au.py` | per-AU agreement figure |
| `scripts/p1a_analyses.py` | per-actor / within-video fidelity, threshold sensitivity, AU bottleneck, tuned MediaPipe baselines |
| `scripts/pipeline_latency.py` | full-pipeline CPU latency (detector + crop, and teacher landmarker) |
| `scripts/export_splits.py` | actor identifiers of every split |

## Data

RAVDESS is **not** redistributed here. Download it from Zenodo
(<https://zenodo.org/records/1188976>, CC BY-NC-SA 4.0) and cite:
S. R. Livingstone and F. A. Russo, *PLOS ONE* 13(5): e0196391, 2018.

## License

Code: MIT (see `LICENSE`). Results tables: CC BY 4.0.
