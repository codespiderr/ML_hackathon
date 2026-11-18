"""Server-only inference and scoring. Reference labels never enter responses."""
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import folium
import joblib
import numpy as np
from sklearn.metrics import f1_score, log_loss

from src.config import BASE_DIR, MODELS_DIR
from src.data_processing.loader import load_trip
from src.data_processing.spatial_utils import prepare_geometry
from src.data_processing.labels import align_reference, load_reference
from src.features.feature_extractor import generate_candidates
from src.route_reconstruction.hmm_decoder import decoder

DATA = Path(os.environ.get('PATHSNAP_EVALUATOR_DIR', str(BASE_DIR / 'results/evaluator_test/data/evaluator_test')))
log = logging.getLogger('uvicorn.error')

def trip_ids():
    return sorted(p.name for p in DATA.iterdir() if p.is_dir() and (p / f'{p.name}.track').exists()) if DATA.exists() else []

@lru_cache(maxsize=1)
def model():
    return joblib.load(MODELS_DIR / 'random_forest_final.joblib')

@lru_cache(maxsize=10)
def process(trip_id):
    if trip_id not in trip_ids():
        raise KeyError(trip_id)
    started = perf_counter()
    directory = DATA / trip_id
    trip = load_trip(directory)
    geometry = prepare_geometry(trip)
    candidates = generate_candidates(geometry)
    shape = candidates.arc_ids.shape
    probabilities = model().predict_proba(candidates.features.reshape(-1, 8))[:, 1].reshape(shape)
    raw = candidates.arc_ids[np.arange(shape[0]), probabilities.argmax(axis=1)]
    settings = json.loads((MODELS_DIR / 'decoder_config.json').read_text())
    predicted = decoder(trip, candidates.arc_ids, probabilities, trip.track[:, 2], geometry.block_starts, **settings)
    inference_seconds = perf_counter() - started
    # Reference is loaded only after predictions are complete, for internal scoring.
    reference = load_reference(directory, len(trip.arcs))
    truth, _, distance = align_reference(geometry, reference)
    labels = candidates.arc_ids == truth[:, None]
    within = np.ones(len(raw) - 1, dtype=bool)
    within[geometry.block_starts[geometry.block_starts > 0] - 1] = False
    def continuity(arcs):
        good = (arcs[:-1] == arcs[1:]) | (trip.arcs[arcs[:-1], 1] == trip.arcs[arcs[1:], 0])
        return float(good[within].mean()) if within.any() else None
    result = dict(trip_id=trip_id, observations=len(raw), model='Tuned Random Forest + Viterbi',
                  top1=float((predicted == truth).mean()), classifier_top1=float((raw == truth).mean()),
                  continuity=continuity(predicted), classifier_continuity=continuity(raw),
                  candidate_recall=float(labels.any(axis=1).mean()),
                  candidate_macro_f1=float(f1_score(labels.ravel(), probabilities.ravel() >= .5, average='macro', zero_division=0)),
                  candidate_log_loss=float(log_loss(labels.ravel(), probabilities.ravel(), labels=[False, True])),
                  inference_seconds=round(inference_seconds, 3),
                  alignment_over_50m=int((distance > 50).sum()),
                  label_note='Top-1 uses approximate reference-to-GPS alignment, scored on the server.')
    route_map = folium.Map(location=trip.track[:, :2].mean(axis=0)[::-1].tolist(), tiles='OpenStreetMap', prefer_canvas=True)
    segments = trip.nodes[trip.arcs]
    low, high = trip.track[:, :2].min(axis=0), trip.track[:, :2].max(axis=0)
    visible = ((segments.max(axis=1) >= low-.002) & (segments.min(axis=1) <= high+.002)).all(axis=1)
    for title, lines, color, weight, show in (
        ('Local road network', segments[visible], '#94a3b8', 1, False),
        ('Forest predictions', segments[np.unique(raw)], '#a855f7', 3, False),
        ('Predicted route · Viterbi', segments[np.unique(predicted)], '#087f8c', 5, True)):
        group = folium.FeatureGroup(name=title, show=show).add_to(route_map)
        folium.GeoJson({'type':'MultiLineString','coordinates':lines.tolist()},
                       style_function=lambda _, c=color, w=weight: {'color':c,'weight':w,'opacity':.85}).add_to(group)
    gps = folium.FeatureGroup(name='GPS observations', show=True).add_to(route_map)
    for lon, lat in trip.track[:, :2]:
        folium.CircleMarker([lat, lon], radius=2, color='#ea7b24', weight=0, fill=True, fill_opacity=.65).add_to(gps)
    for label, point in [('Start', trip.track[0]), ('End', trip.track[-1])]:
        folium.Marker(point[:2][::-1].tolist(), tooltip=label).add_to(route_map)
    folium.LayerControl(collapsed=False).add_to(route_map)
    route_map.fit_bounds([low[::-1].tolist(), high[::-1].tolist()])
    log.info('trip=%s observations=%s top1=%.4f continuity=%.4f inference_seconds=%.3f', trip_id, len(raw), result['top1'], result['continuity'], inference_seconds)
    return result, route_map.get_root().render()
