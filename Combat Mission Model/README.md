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

Team selection appears in the browser at **http://localhost:8000/ui/** — same backend as Swagger; submit the form to see ranked roster options without using `/docs`.

### Docker (train baked in + serves reference API)

```bash
docker build -t combat-mission-model .
docker run -p 8000:8000 combat-mission-model
```

### Smoke-check the HTTP API locally

Requires `pip install -r requirements.txt`. From **this directory** (`Combat Mission Model/`):

```bash
python3 -m unittest discover -v -s tests
```

Confirms lifespan load, `/health`, `/team/select`, rankings, soldier profiles, metadata, OpenAPI link, CORS behaviour, and a 400 path.

The image listens on the `PORT` environment variable (defaults to `8000`) so it works on Render, Railway, Fly.io, Cloud Run, etc.

### Deploy for your team (recommended: Render)

1. Push this repository to GitHub (if it is not already).
2. In [Render](https://render.com), create a **Blueprint** (or **Web Service** → **Docker**) and point it at the repo.
3. If you use the included root `render.yaml`, Render will build from `Combat Mission Model/Dockerfile` with the correct context.
4. After deploy, share the public URL Render assigns (for example `https://combat-mission-api.onrender.com`). Open `/docs` on that host for Swagger UI.
5. From the **repository root**, verify the deployed service responds and allows cross-origin browsers (CORS preflight):

```bash
python3 scripts/verify_live_api.py --base-url https://YOUR-SERVICE.onrender.com
```

If Render returns "no server" (`x-render-routing: no-server`), the web service is not created or not linked to this repo yet—finish the Blueprint / Web Service in the Render dashboard first.

**Teammates** can call the API at `GET /health`, `POST /team/select`, etc. **CORS** is configurable with `CORS_ORIGINS` (default `*` in the blueprint so browsers can call from another origin).

**Security:** This reference API does not implement authentication. Treat the public URL as **open**. For restricted access, front it with SSO/VPN/API keys or deploy to a private network.

Free tiers may **cold-start** after idle periods (first request can be slow).

---

## Roadmap (conceptual)

1. **Model quality** — improve data, features, and validation metrics until predictions are trustworthy on held-out evaluation behavior.
2. **Deploy the artifact** — ship `MODEL_PATH` + metadata to your inference environment (batch or real-time).
3. **Athena (later)** — use model outputs to drive commander-facing insights (gaps, trends, team fit); keep that layer separate from training code here.

---

## Cross-origin browsers (CORS)

The FastAPI stack uses **`CORSMiddleware`** so cross-site frontends receive correct **`Access-Control-Allow-*`** responses and **`OPTIONS` preflight** is answered (**`Allow-Methods`**, **`Allow-Headers`**, **`Max-Age`**, default **`86400` s**).

- **`CORS_ORIGINS=*`** (default in **render.yaml**) — `Access-Control-Allow-Origin: *`. Simple for many teams calling from Postman plus any web app origin. Browsers cannot use **`fetch` with `credentials: "include"`** together with `*` — use bearer tokens without cookies, or list explicit origins instead.
- **Explicit origins** — set **`CORS_ORIGINS=https://app.example.com`** (comma- or whitespace-separated list). Mirrors the calling origin in **`Access-Control-Allow-Origin`**. Optionally **`CORS_ALLOW_CREDENTIALS=true`** (default) for cookie / credentialed requests from those sites; set **`false`** if everything is bearer-token + no cookies.
- **Disable browser CORS** — set **`CORS_ORIGINS=`** (empty).

---

## Environment variables

| Variable | Default (local) | Description |
|----------|-------------------|-------------|
| `EVAL_PATH` | `./evaluations_full.csv` | Evaluations CSV |
| `DETAIL_PATH` | `./detailed_full.csv` | Detailed subtasks CSV |
| `MODEL_PATH` | `./athena.joblib` | Serialized trained pipeline |
| `MODEL_METADATA_PATH` | `./models/mission_model_metadata.json` | Training metadata JSON |
| `PORT` | `8000` | HTTP port (hosts like Render/Railway set this automatically) |
| `CORS_ORIGINS` | `*` | Comma- or whitespace-separated browser origins (e.g. `https://yourapp.vercel.app`); `"*"` allows any origin |
| `CORS_ALLOW_CREDENTIALS` | `true` | When origins are explicit lists, allow cookie / credentialed browser requests. Ignored when `CORS_ORIGINS=*`. |
| `CORS_PREFLIGHT_MAX_AGE` | `86400` | How many seconds browsers may cache `OPTIONS` preflight responses. |

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
