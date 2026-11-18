# PathSnap preprocessing handoff

## Model handoff

```python
from pathlib import Path

import numpy as np

path = Path("results/preprocessing/train/00000001.npz")
with np.load(path) as trip:
    features = trip["features"]      # (observations, candidates, features)
    targets = trip["labels"]        # (observations, candidates), boolean
    arc_ids = trip["arc_ids"]       # (observations, candidates), trip-local IDs
    feature_names = trip["feature_names"]

X = features.reshape(-1, len(feature_names))
y = targets.ravel()
```

Group CV by trip filename before flattening candidate rows. Fit any scaler or
other learned transformation inside each training fold. At evaluation time,
reshape scores to the candidate matrix and select the best arc per observation.
An observation with no positive label has a candidate miss. Include it in the
denominator for end-to-end Top-1 and record candidate recall separately.

For the decoder, use candidate arc IDs, timestamps, the fraction along each
directed arc, and the graph from the original trip. `block_starts` lists
observation indices where decoding can restart. Inference can call
`load_trip`, `prepare_geometry`, and `generate_candidates` from
`src.data_processing.loader`, `src.data_processing.spatial_utils`, and
`src.features.feature_extractor`, respectively, with only `.track`, `.nodes`, and `.arcs`.
The prepared archives also contain reference data for development. Use only
the feature tensor as model input.

## Method and assumptions

- The loader validates shapes, finite values, coordinate ranges, node/arc IDs,
  and strictly increasing timestamps. The geometric approximation converts
  lon/lat to local metres at the median trip latitude.
- Candidate generation finds the 12 closest directed segments by
  point-to-segment distance by default. Ties are ordered by arc ID.
- Features cover distance, directional heading agreement, heading availability,
  local speed, arc length, projected position on the arc, distance rank, and
  distance above the closest candidate. Motion uses neighbouring fixes within
  each GPS block. A gap over 30 seconds starts a new block. Heading is marked
  unavailable for weak motion or a zero-length arc.
- The reference file encodes a sequence of traversed arcs. For training and
  scoring, dynamic programming aligns fixes to nondecreasing positions on
  that reference sequence within each block, minimizing distance plus a
  heading mismatch penalty. This is a proxy for per-fix truth. Report its
  distance diagnostics and interpret pointwise scores alongside sequence
  metrics derived from the original reference route.
- Candidate count, gap threshold, and heading penalty are initial engineering
  choices. Candidate recall and nearest-arc Top-1 in the summaries are measured
  against the alignment proxy.

## Baseline results

The initial preprocessing run used 12 candidate arcs and a 30-second gap threshold.

| Split | Trips | GPS observations | Candidate recall | Nearest-arc Top-1 |
| --- | ---: | ---: | ---: | ---: |
| Training | 80 | 208,417 | 92.80% | 49.42% |
| Student test | 10 | 19,160 | 99.88% | 53.48% |

Training trips `00000033` and `00000054` account for about 91% of training
candidate misses. For missed fixes in those trips, the median distance to
the nearest reference arc was about 133 m and 166 m, respectively. The
training-only candidate count sweep and grouped model training are pending.
