# Random Forest tuning and route decoding

Run from the repository root with the dependencies in `requirements.txt`:

```powershell
python -m src.models.bayesian_opt --trials 30 --search-cap 12000 --final-cap 60000
python -m src.route_reconstruction.evaluate_decoder
```

The Optuna TPE study searches five Random Forest parameters. Its objective is
mean observation-level Top-1 accuracy across the five saved trip folds in
`results/model_comparison/folds.json`. Each trial samples at most 12,000
training GPS fixes per fold, keeping all candidate rows for each sampled fix.
The final comparison fits at most 60,000 fixes sampled across all 80 training
trips. The student-test split is evaluated after tuning.

The 30-trial best configuration was 200 trees, depth 10, minimum leaf size 6,
maximum feature fraction 0.8144, and bootstrap sample fraction 0.6502.
The best search score was 0.74564 mean CV Top-1. With the same 60,000-fix cap,
the subsequent five-fold comparison measured 0.74446 for the original
configuration and 0.74622 for the tuned configuration. The difference between
the search best score and later comparison reflects the different training
caps.

| Student-test metric | Untuned | Tuned |
| --- | ---: | ---: |
| Top-1 accuracy | 0.84870 | 0.85063 |
| Candidate-row macro F1 | 0.91307 | 0.91544 |
| Candidate-row log loss | 0.06547 | 0.06408 |
| Direct route continuity | 0.92203 | 0.92239 |
| Candidate-row inference time on 10 student-test trips | 0.17 s | 0.29 s |

Isotonic calibration was fitted on a held-out training fold. It reduced log
loss on the fold used to fit the calibrator, but student-test log loss rose
from 0.06408 to 0.08265. The final saved model therefore uses the tuned
forest's raw probabilities. The calibration fold number, raw and calibrated
results, trial history, and reliability plot are recorded under
`results/random_forest_tuning/`.

The decoder uses candidate road arcs as hidden states. It normalises classifier
probabilities at each GPS fix for emission scores. Transition scores compare
the directed network path length with observed GPS displacement. Disconnected
candidate pairs receive a finite penalty. Decoding restarts after the GPS
gaps marked in the prepared archives. Decoder settings were chosen using two
held-out training trips, then evaluated on all ten student-test trips.
The decoder results JSON records overall and per-trip pointwise accuracy,
direct route continuity, and reference transition F1. It records trips where
decoding lowered pointwise accuracy.

On student-test, Viterbi decoding changed Top-1 accuracy from 0.85063 to
0.86608 and direct continuity from 0.92239 to 0.96642. Mean per-trip
reference transition F1 rose from 0.64373 to 0.74757. Pointwise accuracy
fell on three trips, including trip `00000065`, where it fell by 0.00651.

The student-test results are development-stage measurements. Pointwise labels
come from the earlier monotonic alignment of GPS fixes to the reference
route, so they are approximate.
