# SwiftParcel ETA — Assignment 5

Krishanthh Mohan · DA24B044

The required implementations and measured evidence are in `partA`–`partD`. `REPORT.pdf` contains all tables, written answers, machine details, abort criteria and the list of items not tested. The demo video is kept separately in the repository's `assignment videos` folder.

## Setup

Run commands from this folder. Python 3.11 is the validated version; CPU only.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MLFLOW_ENABLE_TELEMETRY=false RAY_USAGE_STATS_ENABLED=0
python eta_data.py
python -m pytest -q
```

`eta_data.py` regenerates the supplied deterministic training/fresh data and both bundles. Models, datasets and databases are intentionally excluded from submission. The 34 tests need these generated bundles. Use a writable local directory for the generated data and MLflow artifacts.

## A — REST service and measurements

```bash
python -m uvicorn partA.eta_app:app --host 127.0.0.1 --port 8005 --workers 1
```

Open `http://localhost:8005/docs` for the interactive API. Endpoints: `POST /v1/models/eta:predict`, `GET /healthz`, `GET /metrics`. `ETA_MODEL` can point to a different bundle; restart the service after changing it. Every worker loads its own model once. Prometheus counters are per worker; scrape a one-worker instance when reproducing the basic metrics checks.

Example body:

```json
{"instances":[{"distance_km":20,"n_stops":5,"hour":12,"weekday":2,"traffic_index":0.3,"rain":0,"weight_kg":8}]}
```

In a second activated terminal, create the load-client sample and measure:

```bash
python - <<'PY'
import pandas as pd
from eta_data import FEATURES
d = pd.read_csv('data/eta_fresh.csv')
d = d[~((d.distance_km > 150) & (d.n_stops > 20))]
d[FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)
PY
python load_test.py --url http://127.0.0.1:8005/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 --out partA/loadtest_results.json
python load_test.py --url http://127.0.0.1:8005/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 1 --out /tmp/eta-batch1.json
python load_test.py --url http://127.0.0.1:8005/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 100 --out /tmp/eta-batch100.json
```

For the measured change, stop the single-worker server and run the same Uvicorn command with `--workers 2`; repeat the five concurrency measurements. `partA/additional_measurements.json` retains the completed two-worker and batching measurements. The baseline report used one warm-up before timed traffic.

## B — Registry, promotion gate and shadow

```bash
export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
python partB/eta_mlflow.py register
python partB/eta_mlflow.py gate
python partB/eta_mlflow.py gate --min-rel-gain 0.60
```

Register once on a clean database to obtain registry versions 1/2 and aliases `prod`/`candidate`; registering again creates new versions. The completed registry table and both gate outputs are in `partB/gate_output.txt`. Gate inference timing is distinct from shadow HTTP timing.

In two activated terminals with the same tracking URI:

```bash
mlflow models serve --model-uri models:/eta-model@prod --host 127.0.0.1 --port 5101 --workers 1 --env-manager local
mlflow models serve --model-uri models:/eta-model@candidate --host 127.0.0.1 --port 5102 --workers 1 --env-manager local
```

Run each server command separately and keep both alive. Then execute:

```bash
python partB/eta_mlflow.py shadow > partB/shadow_output.txt
```

Optional registry UI: `mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI" --host 127.0.0.1 --port 5050`.

## C — Two-CPU sticky canary

Stop the MLflow scoring servers before this phase on a low-memory machine. Port 8055 and dashboard port 8265 must be available; do not connect another Ray cluster to this verification.

```bash
python partC/verify_ray.py
# Or keep the verified app live for interactive requests:
python partC/verify_ray.py --keep-alive
```

The script creates a real two-CPU Ray runtime, runs all five required assertions, writes `partC/verify_output.txt`, and cleans up. `--keep-alive` leaves the final 10% canary running until Ctrl+C. Its prediction body adds `"customer_id":"customer-0001"` to the example REST body. `X-Beta: 1` forces v2 for that request. The public `partC.eta_ray.build(weight=0.1, extra_work_ms=0.0)` returns a Ray Serve application; deployment CPU reservations fit two CPUs even at maximum candidate replicas.

## D — Release smoke gate

With the Part A v1 service running:

```bash
python partD/smoke_eta.py --url http://127.0.0.1:8005 --expect-version v1 --max-p95-ms 80
```

Exit 0 means all eight checks passed; exit 1 rejects release. `partD/smoke_output.txt` records the completed release verification. The report gives four proposed canary abort rules.

## Source and assistance disclosure

`eta_data.py` is the supplied assignment starter, reused unchanged. `load_test.py` is adapted from the supplied client to add actual elapsed time, mean latency and batch cost per row. See `AI_DISCLOSURE.md` for the scope of AI assistance.
