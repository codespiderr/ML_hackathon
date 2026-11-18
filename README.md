# PathSnap: Distributed Map-Matching & Route Reconstruction

End-to-end Machine Learning and Hidden Markov Model (HMM) system for matching noisy GPS trajectory observations to road networks and reconstructing true vehicle paths.

---

## 1. Challenge & Problem Overview
GPS tracking signals in urban canyons suffer from multipath reflection, building shadows, sensor noise, and variable sampling intervals. Consequently, recorded coordinates frequently deviate onto adjacent streets, jump across intersections, or deviate into buildings.

**PathSnap** tackles this challenge using:
1. **Spatial & Road Network Indexing**: Exact nearest-segment lookup using a spatial grid.
2. **Feature Engineering**: Distance, heading agreement, speed, and position along the road segment.
3. **Multi-Model Machine Learning Benchmark**: Supervised classification comparing $\ge 5$ model families (Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, MLP / KNN).
4. **Bayesian Hyperparameter Optimisation**: Group-aware cross-validated hyperparameter tuning using Optuna.
5. **HMM & Viterbi Sequential Decoding**: Global trajectory optimization enforcing network road connectivity and distance plausibility.
6. **Distributed Two-Laptop Deployment**:
   - **Laptop 1 (Model & Backend Server)**: FastAPI REST API handling inference requests, loading serialized models, and computing server-side evaluation.
   - **Laptop 2 (Evaluator UI)**: Interactive dashboard (Streamlit) for evaluators to trigger trip evaluation and inspect interactive multi-layer Folium maps and metrics.

---

## 2. Directory Structure

```
├── api/                           # Laptop 1: FastAPI REST API backend
│   ├── main.py                    # API endpoints (/predict, /evaluate, /trips, /map)
│   └── schemas.py                 # Pydantic data schemas
├── ui/                            # Laptop 2: Evaluator User Interface
│   ├── app.py                     # Streamlit web application
│   └── components.py              # Visual components and Folium map embeds
├── src/                           # Core algorithms & pipeline
│   ├── config.py                  # Global configurations, hyperparams & paths
│   ├── data_processing/           # Trajectory & road-network graph ingestion
│   │   ├── loader.py              # Parses .track, .nodes, and .arcs
│   │   ├── labels.py              # Reads .route and aligns it to GPS observations
│   │   ├── prepare.py             # Writes candidate arrays and EDA summaries
│   │   └── spatial_utils.py       # Local metric projection and segment geometry
│   ├── features/                  # Candidate generation & feature extraction
│   │   ├── candidate_gen.py       # Indexed nearest-segment search
│   │   └── feature_extractor.py   # Distance, heading error, speed, projection features
│   ├── models/                    # Model training, evaluation & tuning
│   │   ├── train_models.py        # Trains 5+ distinct algorithm families
│   │   ├── evaluate.py            # Top-1 accuracy, Macro F1, Log Loss, Confusion Matrix
│   │   └── bayesian_opt.py        # Optuna Bayesian hyperparameter search
│   ├── route_reconstruction/      # Sequential map-matching
│   │   ├── hmm_decoder.py         # Emission & transition modeling + Viterbi algorithm
│   │   └── continuity.py          # Sequence-level metrics & path continuity checks
│   └── visualization/             # Interactive maps and EDA plotting
│       ├── map_visualizer.py      # Multi-layer Folium maps with layer controls
│       └── eda_plots.py           # Distribution, trajectory & model comparison plots
├── saved_models/                  # Serialized trained models & metadata
├── results/                       # Evaluation CSVs, metrics, and generated figures
│   └── plots/
├── maps/                          # Exported interactive HTML Folium maps
├── report/                        # Technical report materials & figures
├── requirements.txt               # Python package dependencies
└── README.md                      # Documentation & instructions
```

---

## 3. Installation & Quickstart

### Environment Setup
```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Execution Pipeline
1. **Preprocess the supplied development trips**:
   ```bash
   python -m src.data_processing.prepare
   ```
   The checkout includes the 80 training trips and 10 student-test trips in
   `Student folder/Student folder`. Each trip has its original `.track`,
   `.nodes`, `.arcs`, and `.route` files. The reserved evaluator trips are
   supplied separately. Each processed split writes one candidate archive per
   trip and a JSON summary under `results/preprocessing/`. Use `--data` to
   point at a different copy. For a quick check:
   ```powershell
   python -m src.data_processing.prepare --split train --trip 00000001
   ```
   See [the preprocessing handoff](docs/preprocessing.md) for file shapes and
   alignment assumptions. The following stages are planned in the repository scaffold.
2. **Model Training & Comparison (5+ Models)**:
   ```bash
   python -m src.models.train_models
   ```
3. **Bayesian Hyperparameter Tuning**:
   ```bash
   python -m src.models.bayesian_opt
   ```
4. **HMM Route Reconstruction**:
   ```bash
   python -m src.route_reconstruction.hmm_decoder
   ```

---

## 4. Two-Laptop Deployment Architecture

### Laptop 1: API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API docs available at `http://localhost:8000/docs`.

### Laptop 2: Evaluator Dashboard
Set `API_BASE_URL` in `ui/app.py` or through the UI to Laptop 1's IP address (e.g., `http://192.168.x.x:8000`).
```bash
streamlit run ui/app.py --server.port 8501
```
Evaluators can select any of the 10 evaluation trips, trigger inference, and inspect ground-truth vs reconstructed routes with detailed metrics.
