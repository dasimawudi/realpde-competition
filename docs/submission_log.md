# Submission Log

## Codabench successful submissions

| ID | File | Date | Final | rel_l2 | TKE | MVPE | Time | SPS | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 897948 | `submission_cno_baseline.zip` | 2026-08-23 13:37 | 64.52290 | 88.581214 | 65.718217 | 81.677206 | 88.663276 | 5.192310 | Official CNO baseline package. |
| 900896 | `submission_cno_realft_4700_20260825.zip` | 2026-08-25 15:27 | 70.04628 | 93.258780 | 68.856072 | 92.016287 | 88.618744 | 10.578370 | Real-finetuned CNO. |
| 903976 | `submission_cno_tke1200_bounds_rel00.zip` | 2026-08-27 15:44 | 75.58455 | 93.542062 | 70.934325 | 92.167656 | 87.236663 | 27.780536 | Current best known Codabench result. |
| 907047 | `8-29提交.zip` | 2026-08-29 13:47 | 74.48384 | 91.868766 | 66.666667 | 89.885887 | 91.959120 | 27.374631 | UNet local-proxy candidate; hidden physical scores worse than CNO. |
| TBD | `submission_cno_tke4100_lam215_microa020_abs0075_rel0075_nobench_20260830.zip` | 2026-08-30 | 75.94193 | TBD | TBD | TBD | TBD | TBD | User-reported hidden score; current score-to-beat baseline for feature-engineering runs. |

## Local packages prepared on 2026-08-29

These are flat-clean CNO `tke4100` single-model candidates. They were smoke-tested locally on the remote machine and downloaded to the local workspace.

| File | Role |
|---|---|
| `submission_cno_tke4100_bounds_abs0075_rel000_flat_20260829.zip` | Recommended next one-shot candidate; closest to the known good CNO `rel00` route. |
| `submission_cno_tke4100_bounds_abs0075_rel010_flat_20260829.zip` | Local SPS proxy best among scanned simple bounds. |
| `submission_cno_tke4100_bounds_abs0075_rel020_flat_20260829.zip` | Matches earlier CNO package style with relative bound. |

Additional CNO-only low-learning-rate continuation from `tke4100`:

| File | Role |
|---|---|
| `submission_cno_tke4100_cont600_balanced_abs0075_rel000_flat_20260829.zip` | More aggressive CNO candidate; local TKE/MVPE improved, Rel-L2 slightly worse. |
| `submission_cno_tke4100_cont600_balanced_abs0075_rel010_flat_20260829.zip` | Same checkpoint with local simple-bound variant. |
| `submission_cno_tke4100_cont600_balanced_abs0075_rel020_flat_20260829.zip` | Same checkpoint with relative bound variant. |

Local continuation best summary:

```text
run: cno_tke4100_cont_lr5e7_balanced_20260829
best_iter: 600
rel_l2: 0.10214869
tke: 0.71349053
mvpe: 0.10154706
local_mean5_proxy: 79.09720
```

Single-model interpolation/extrapolation between original `tke4100` and `cont600`:

| File | Role |
|---|---|
| `submission_cno_tke4100_continterp_lam125_abs0075_rel000_flat_20260829.zip` | Aggressive CNO-only candidate, using lambda=1.25 and rel=0.0 bounds. |
| `submission_cno_tke4100_continterp_lam125_abs0075_rel010_flat_20260829.zip` | Local best among the CNO-only candidates; recommended if explicitly trying to beat current score. |
| `submission_cno_tke4100_continterp_lam125_abs0075_rel020_flat_20260829.zip` | Same checkpoint with relative bound variant. |

Local interpolation best summary:

```text
run: cno_tke4100_to_cont600_weight_interp_scan_20260829
lambda_cont600: 1.25
rel_l2: 0.10238736
tke: 0.70794994
mvpe: 0.10147392
local_mean5_proxy: 79.16723
best_local_bound: abs=0.0075, rel=0.01
```

## Lessons learned

- Do not optimize the self-written equal-weight final estimate. Codabench states the `final_score` combination is not published.
- UNet postprocessing can look strong on released validation data but generalize poorly on hidden data.
- CNO currently has better hidden physical scores, especially Rel-L2, TKE, and MVPE.
- Prefer simple CNO packages until a new candidate improves hidden physical subscores.

## Local packages prepared on 2026-08-30

The 2026-08-30 work focused on CNO-only weight-space moves, because the 2026-08-29 UNet submission underperformed on hidden physical metrics.

Important scoring note: the current Codabench announcement says Track 1 is on Starting Kit v9, where `sps_score` is mapped linearly and `final_score` combination is not published. A temporarily downloaded older v6 `scoring.py` was useful for validity checks but not for choosing the final score.

### Weight extrapolation from `tke4100` to `cont600`

Scanned `lambda_cont600` values around the previous best direction. The best region was broad and shallow around `2.1` to `2.35`; higher extrapolation started to hurt Rel-L2/SPS.

Representative local validation rows:

| Variant | rel_l2 | TKE | MVPE | Bound | Note |
|---|---:|---:|---:|---|---|
| `lambda=2.15` | 0.103102 | 0.695704 | 0.101154 | `abs=0.0075, rel=0.0075` | Safer than 2.35; better Rel/SPS. |
| `lambda=2.35` | 0.103898 | 0.694675 | 0.101337 | `abs=0.0075, rel=0.0075` | Slightly better TKE; more extrapolated. |

Generated packages:

| File | Role |
|---|---|
| `submission_cno_tke4100_cont600_extrapolate_lam215_abs0075_rel0075_20260830.zip` | Pure extrapolation, safer side of the peak. |
| `submission_cno_tke4100_cont600_extrapolate_lam235_abs0075_rel0075_20260830.zip` | Pure extrapolation, TKE-heavy side of the peak. |

### Rel/MVPE micro-tune and interpolation

A low-LR micro-tune from `lam215` improved Rel-L2/MVPE/SPS but gradually gave back TKE. The best saved checkpoint was at 200 updates.

Interpolating between pure `lam215` and the 200-step micro-tuned checkpoint found the best local tradeoff at `alpha_micro=0.20`.

Final recommended package:

| File | Role |
|---|---|
| `submission_cno_tke4100_lam215_microa020_abs0075_rel0075_nobench_20260830.zip` | Current main candidate. Single CNO model; `lambda_cont600=2.15`, then 20% mix toward Rel/MVPE micro-tune. Uses `abs=0.0075, rel=0.0075` bounds and disables `cudnn.benchmark` to avoid cold-start timing spikes. |

Local validation summary for the final candidate:

```text
base: cno_tke4100_cont600_extrapolate_lam215_20260830.pth
micro target: cno_lam215_rel_mvpe_micro_lr1e7_20260830/model_best.pth
alpha_micro: 0.20
rel_l2: 0.10305153
tke: 0.69582664
mvpe: 0.10112838
best_bound: abs=0.0075, rel=0.0075
zip: submission_cno_tke4100_lam215_microa020_abs0075_rel0075_nobench_20260830.zip
```

Packaging check:

```text
model.pth
rpde_baselines/__init__.py
rpde_baselines/cno.py
submission.py
```

## Local packages prepared on 2026-08-31

Work moved to the `192.168.0.148` GPU server with data under `/home/chyfuture/RealPDE_data/p0ab_real_h5_20260830/`. The server has an RTX 3090 24GB and a reusable Docker image with PyTorch `2.2.2+cu121`; `h5py` and `matplotlib` were installed into a temporary local training image.

Important HDF5 split note:

```text
usable h5 files: 81 after excluding 7575_0.h5
train trajectories: 65
validation trajectories: 16
train windows: 2701
validation windows: 640
```

Negative experiments:

- Frozen residual feature adapter degraded quickly even at low learning rate. It is not ready to submit.
- Low-learning-rate continuation from the current best CNO also degraded on the HDF5 validation split by 100 updates.
- Interpolating the current best CNO toward the existing P0 baseline checkpoint degraded sharply even at `alpha=0.005`; negative extrapolation produced NaNs.

Bounds-only candidate:

| File | Role |
|---|---|
| `submission_cno_tke4100_lam215_microa020_abs0075_rel015_nobench_20260831.zip` | Same prediction model as the 75.94193 submission, but with `abs=0.0075, rel=0.015` bounds based on the 2026-08-31 HDF5 validation sweep. |

HDF5 validation summary for current-best prediction with fine bounds:

```text
checkpoint: current_best/model.pth extracted from submission_cno_tke4100_lam215_microa020_abs0075_rel0075_nobench_20260830.zip
rel_l2: 0.11486314
tke: 0.66078759
mvpe: 0.10656577
best_bound: abs=0.0075, rel=0.015
local_mean5_proxy: 77.71761
zip: submission_cno_tke4100_lam215_microa020_abs0075_rel015_nobench_20260831.zip
```

Residual-correction candidate:

| File | Role |
|---|---|
| `submission_cno_residualcorr_h24_b2_step200_alpha100_abs0075_rel015_20260831.zip` | Freezes the current best CNO and adds a tiny feature-driven 3D-conv residual corrector. Uses `alpha=1.0`, `abs=0.0075`, `rel=0.015`. |

Full HDF5 validation summary for residual correction:

```text
checkpoint: residual_corr_fast_h24_b2_lr1e4_md004_20260831_1535/model_best.pth
train: 200 updates, hidden=24, blocks=2, max_delta=0.04, residual_mse=0.15, delta_penalty=0.05
best_alpha: 1.0
best_bound: abs=0.0075, rel=0.015
rel_l2: 0.11126853
tke: 0.66070499
mvpe: 0.10258214
local_mean5_proxy: 77.76902
zip: submission_cno_residualcorr_h24_b2_step200_alpha100_abs0075_rel015_20260831.zip
```
