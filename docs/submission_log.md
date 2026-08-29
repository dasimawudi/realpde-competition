# Submission Log

## Codabench successful submissions

| ID | File | Date | Final | rel_l2 | TKE | MVPE | Time | SPS | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 897948 | `submission_cno_baseline.zip` | 2026-08-23 13:37 | 64.52290 | 88.581214 | 65.718217 | 81.677206 | 88.663276 | 5.192310 | Official CNO baseline package. |
| 900896 | `submission_cno_realft_4700_20260825.zip` | 2026-08-25 15:27 | 70.04628 | 93.258780 | 68.856072 | 92.016287 | 88.618744 | 10.578370 | Real-finetuned CNO. |
| 903976 | `submission_cno_tke1200_bounds_rel00.zip` | 2026-08-27 15:44 | 75.58455 | 93.542062 | 70.934325 | 92.167656 | 87.236663 | 27.780536 | Current best known Codabench result. |
| 907047 | `8-29提交.zip` | 2026-08-29 13:47 | 74.48384 | 91.868766 | 66.666667 | 89.885887 | 91.959120 | 27.374631 | UNet local-proxy candidate; hidden physical scores worse than CNO. |

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

## Lessons learned

- Do not optimize the self-written equal-weight final estimate. Codabench states the `final_score` combination is not published.
- UNet postprocessing can look strong on released validation data but generalize poorly on hidden data.
- CNO currently has better hidden physical scores, especially Rel-L2, TKE, and MVPE.
- Prefer simple CNO packages until a new candidate improves hidden physical subscores.
