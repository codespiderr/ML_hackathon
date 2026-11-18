# Two-device demonstration

## Laptop 1: model server

Run from the ML_hackathon directory:

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_EVALUATOR_DIR = 'C:\absolute\path\to\evaluator_test'
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The configured folder must contain the ten trip directories, each with .track,
.nodes, .arcs and .route files. The default uses the extracted evaluator archive
under results/evaluator_test/data/evaluator_test. Keep reference files on this
device. The server loads saved_models/random_forest_final.joblib and
saved_models/decoder_config.json, never the experimental boosted model.

Use ipconfig to find this laptop's Wi-Fi IPv4 address. Both devices must be on
the same network. If Windows prompts, allow Python on the private network.
If a firewall blocks the port, allow TCP 8000 on the private network only.
Keep the server terminal visible: Uvicorn logs requests and the inference
service logs trip ID, observation count, accuracy, continuity and duration.

## Laptop 2: client

Copy only ui/app.py and ui/requirements.txt to this laptop:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501
```

Open http://localhost:8501 and enter http://SERVER_LAN_IP:8000 in the sidebar.
Select any trip and click Run map matching. The client needs no datasets or
model weights. Internet access is needed for Folium/Leaflet assets and map
tiles. GPS and predicted geometry come from the actual trip; predicted arcs
are displayed as separate segments so gaps are not fabricated as straight lines.

## API and validation

- GET /health: connectivity and trip count.
- GET /trips: available evaluator IDs.
- POST /predict with {"trip_id":"00000015"}: metrics and interactive map.
- POST /evaluate: metrics only.
- GET /map/00000015: map HTML.
- GET /docs: interactive API documentation.

First requests run the complete pipeline; subsequent requests for the same trip
reuse an in-memory result. Restart the server after changing model/data/config.
Reference labels are loaded after inference and never returned to the client.
Top-1 uses approximate monotonic reference alignment. Candidate F1/log loss
describe the classifier, while Top-1 and continuity also report decoded output.

Run python -m api.check_integration to check all ten actual inference requests.
Run python -m streamlit run ui/app.py locally for a same-machine smoke test.
Separate-device connectivity must also be demonstrated on the actual network.
