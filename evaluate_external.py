"""Evaluate the saved model and decoder on a separate archive of trip files."""
import hashlib
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np

from src.data_processing.loader import load_trip
from src.data_processing.spatial_utils import prepare_geometry
from src.data_processing.labels import align_reference, load_reference
from src.features.feature_extractor import generate_candidates
from src.route_reconstruction.hmm_decoder import decoder
from src.route_reconstruction.evaluate_decoder import transition_f1

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / 'evaluator_test (1).zip'
OUT = ROOT / 'results' / 'evaluator_test'
OUT.mkdir(parents=True, exist_ok=True)
data_dir = OUT / 'data'
data_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(ARCHIVE) as archive:
    for info in archive.infolist():
        target = (data_dir / info.filename).resolve()
        if not target.is_relative_to(data_dir.resolve()):
            raise ValueError('Unsafe path in evaluator archive')
    archive.extractall(data_dir)

model_path = ROOT / 'saved_models' / 'random_forest_final.joblib'
model = joblib.load(model_path)
settings = json.loads((ROOT / 'saved_models' / 'decoder_config.json').read_text())
rows = []
for directory in sorted((data_dir / 'evaluator_test').iterdir()):
    if not directory.is_dir():
        continue
    trip = load_trip(directory)
    geometry = prepare_geometry(trip, 30)
    candidates = generate_candidates(geometry, 12)
    ids = candidates.arc_ids
    n, k, features = candidates.features.shape
    scores = model.predict_proba(candidates.features.reshape(-1, features))[:, 1].reshape(n, k)
    raw = ids[np.arange(n), scores.argmax(axis=1)]
    decoded = decoder(trip, ids, scores, trip.track[:, 2], geometry.block_starts, **settings)
    # Load reference only after generating predictions.
    reference = load_reference(directory, len(trip.arcs))
    aligned, _, distance = align_reference(geometry, reference)
    within = np.ones(n - 1, dtype=bool)
    within[geometry.block_starts[geometry.block_starts > 0] - 1] = False
    def connected(pred):
        return int((((pred[:-1] == pred[1:]) | (trip.arcs[pred[:-1], 1] == trip.arcs[pred[1:], 0])) & within).sum())
    row = dict(trip_id=directory.name, observations=n,
               candidate_hits=int((ids == aligned[:, None]).any(axis=1).sum()),
               raw_hits=int((raw == aligned).sum()), decoded_hits=int((decoded == aligned).sum()),
               pairs=int(within.sum()), raw_connected=connected(raw), decoded_connected=connected(decoded),
               raw_transition_f1=transition_f1(raw, reference),
               decoded_transition_f1=transition_f1(decoded, reference),
               alignment_over_50m=int((distance > 50).sum()),
               alignment_p95_m=float(np.percentile(distance, 95)))
    rows.append(row)
    (OUT / 'partial_results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f"{directory.name}: {n} points, raw {row['raw_hits']/n:.2%}, decoded {row['decoded_hits']/n:.2%}", flush=True)

total = lambda key: sum(row[key] for row in rows)
result = dict(archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
              model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
              decoder_settings=settings, trip_count=len(rows), observations=total('observations'),
              candidate_recall=total('candidate_hits')/total('observations'),
              raw_top1=total('raw_hits')/total('observations'),
              decoded_top1=total('decoded_hits')/total('observations'),
              raw_continuity=total('raw_connected')/total('pairs'),
              decoded_continuity=total('decoded_connected')/total('pairs'),
              alignment_over_50m=total('alignment_over_50m'), trips=rows)
(OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'trips'}, indent=2), flush=True)
