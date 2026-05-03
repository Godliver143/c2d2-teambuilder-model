# Combat Mission Model

**Repository:** [github.com/danielyeboah-nu/c2d2-teambuilder-model](https://github.com/danielyeboah-nu/c2d2-teambuilder-model)

## Focus: train, evaluate, deploy the model

The **product of this repo** is a **trained mission-performance model** saved as a joblib artifact, plus a **JSON metadata** sidecar for versioning and integration. Metrics (cross-validated MAE on event scores) are produced every time you run training.

**Athena** (commander communication, “where to improve,” narrative coaching) is intentionally **out of scope here**. Once prediction quality is acceptable, Athena—or any other platform—can **consume** the model’s outputs (scores, rankings, gaps vs peers) over your own APIs.

### Artifacts

| Output | Env override | Purpose |
|--------|----------------|---------|
| Trained bundle (`Athena` pickle) | `MODEL_PATH` | Scorer + soldier profiles + team search for inference |
| `models/mission_model_metadata.json` | `MODEL_METADATA_PATH` | Features, CV MAE, data stats, mission classes — for registries / non-Python consumers |

### Train locally

```bash
pip install -r requirements.txt
python train.py
```

Defaults expect `evaluations_full.csv` and `detailed_full.csv` next to `model.py`. Override with `EVAL_PATH` and `DETAIL_PATH` if needed.

### Optional reference API

`main.py` exposes a small FastAPI app so you can smoke-test predictions during integration. It is **not** the Athena product.

```bash
python train.py
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs

### Docker (train baked in + serves reference API)

```bash
docker build -t combat-mission-model .
docker run -p 8000:8000 combat-mission-model
```

---

## Roadmap (conceptual)

1. **Model quality** — improve data, features, and validation metrics until predictions are trustworthy on held-out evaluation behavior.
2. **Deploy the artifact** — ship `MODEL_PATH` + metadata to your inference environment (batch or real-time).
3. **Athena (later)** — use model outputs to drive commander-facing insights (gaps, trends, team fit); keep that layer separate from training code here.

---

## Environment variables

| Variable | Default (local) | Description |
|----------|-------------------|-------------|
| `EVAL_PATH` | `./evaluations_full.csv` | Evaluations CSV |
| `DETAIL_PATH` | `./detailed_full.csv` | Detailed subtasks CSV |
| `MODEL_PATH` | `./athena.joblib` | Serialized trained pipeline |
| `MODEL_METADATA_PATH` | `./models/mission_model_metadata.json` | Training metadata JSON |

---

## Legacy API table (reference server)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service index (links to docs, health, metadata) |
| GET | `/health` | Model status, resolved artifact paths, metadata flag |
| GET | `/model/metadata` | Training metadata JSON (registry / contracts) |
| GET | `/mission-types` | Canonical mission types |
| GET | `/mission-context/enums` | Allowed context enums |
| POST | `/team/select` | Up to N team options (default 2) + pool breakdowns; optional mission context |
| GET | `/soldiers/rankings/{mission_type}` | Rankings (model scores only; no context modifiers) |
| POST | `/soldiers/rankings` | Rankings with optional `mission_context` (adjusted vs raw scores) |
| GET | `/soldiers` | All profiles |
| GET | `/soldiers/{leader_identifier}` | One profile |

**Client contract:** `POST /team/select` returns `TeamSelectResponse` with required fields `num_team_options_requested` and `num_team_options_returned` (OpenAPI `/docs`). Any client that only parsed the older shape should read these as well.

Canonical mission type labels include: `ambush`, `attack`, `break_contact`, `movement_to_contact`, `react_idf`, `react_uas`, `recon`.
