"""Generate an interactive development route map and a print figure."""
import json

import folium
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from src.config import DATA_DIR, MAPS_DIR, MODELS_DIR
from src.data_processing.loader import load_trip
from src.data_processing.labels import load_reference
from src.data_processing.spatial_utils import prepare_geometry
from src.features.feature_extractor import generate_candidates
from src.route_reconstruction.hmm_decoder import decoder


def main():
    trip_id = "00000052"
    directory = DATA_DIR / "student_test" / trip_id
    trip = load_trip(directory)
    geometry = prepare_geometry(trip)
    candidates = generate_candidates(geometry)
    model = joblib.load(MODELS_DIR / "random_forest_final.joblib")
    p = model.predict_proba(candidates.features.reshape(-1, candidates.features.shape[-1]))
    settings = json.loads((MODELS_DIR / "decoder_config.json").read_text())
    predicted = decoder(
        trip, candidates.arc_ids, p[:, 1].reshape(candidates.arc_ids.shape),
        trip.track[:, 2], geometry.block_starts, **settings,
    )
    reference = load_reference(directory, len(trip.arcs))
    gps = trip.track[:, :2]
    segments = trip.nodes[trip.arcs]
    low, high = gps.min(axis=0), gps.max(axis=0)
    padding = np.maximum((high - low) * 0.12, 0.00015)
    low, high = low - padding, high + padding
    visible = ((segments.max(axis=1) >= low) & (segments.min(axis=1) <= high)).all(axis=1)
    layers = [
        ("Local road network", segments[visible], "#899097", 1),
        ("Reference arcs", segments[np.unique(reference)], "#b34b15", 5),
        ("Viterbi selected arcs", segments[np.unique(predicted)], "#005baa", 2),
    ]
    route_map = folium.Map(location=gps.mean(axis=0)[::-1].tolist(), tiles=None, zoom_snap=0.25)
    folium.TileLayer("OpenStreetMap", show=False).add_to(route_map)
    for name, lines, color, width in layers:
        group = folium.FeatureGroup(name=name).add_to(route_map)
        for line in lines:
            folium.PolyLine(line[:, ::-1].tolist(), color=color, weight=width).add_to(group)
    points = folium.FeatureGroup(name="Raw GPS fixes").add_to(route_map)
    for lon, lat in gps:
        folium.CircleMarker(
            [lat, lon], radius=1.5, weight=0, fill=True,
            fill_color="#242424", fill_opacity=0.6,
        ).add_to(points)
    folium.LayerControl(collapsed=False).add_to(route_map)
    route_map.fit_bounds([low[::-1].tolist(), high[::-1].tolist()])
    route_map.save(MAPS_DIR / f"student_test_{trip_id}.html")

    fig, ax = plt.subplots(figsize=(8.0, 4.5), layout="constrained")
    for name, lines, color, width in layers:
        ax.add_collection(LineCollection(lines, colors=color, linewidths=width * .65, label=name))
    ax.scatter(*gps.T, s=4, color="#242424", label="Raw GPS fixes", zorder=5)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
           xlabel="Longitude (degrees)", ylabel="Latitude (degrees)", title=f"Student-test trip {trip_id}")
    ax.set_aspect(1 / np.cos(np.deg2rad(gps[:, 1].mean())))
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(MAPS_DIR / f"student_test_{trip_id}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
