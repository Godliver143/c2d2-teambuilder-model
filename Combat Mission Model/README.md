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

Easier (always starts in the folder that owns `main.py`):

```bash
./run_server.sh
```

Open **http://127.0.0.1:8000/** — landing page lists **Team selection**, **Swagger**, **Health** (avoid typing `/ui` manually).

Open http://127.0.0.1:8000/docs directly if you prefer Swagger first.

Team selection UI (same app; use **`http://`**, not **`https://`**):

- **http://127.0.0.1:8000/ui** (or `/ui/` with trailing slash)
- **http://127.0.0.1:8000/viewer**
- **http://127.0.0.1:8000/team-selection**

If you see **`{"detail":"Not Found"}`**, wrong path or old server — use links from **http://127.0.0.1:8000/** **after `./run_server.sh`**.

After updating code: **`git pull`**, rerun **`run_server.sh`**, ensure **`web/index.html`** lives next to **`main.py`**.

Use **`http://127.0.0.1:8000/browser-check`** — plain text starting with **OK** confirms you are on **`http://`**.

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

### Deploy for your team (Render → public HTTPS URL)

Render puts the API on **`https://<something>.onrender.com`**. Anyone with the link can hit it (**no auth** in this app).

**Before you start:** Push this repo’s **`main`** branch to GitHub. You need **`render.yaml`** at the repo root, **`Combat Mission Model/requirements-docker.txt`**, **`athena.joblib`**, and **`Combat Mission Model/Dockerfile`** (Dockerfile is optional for Render but kept for container deploys elsewhere).

---

**Option A — Blueprint (recommended)**

1. Sign up / log in at [render.com](https://render.com) and **connect GitHub**.
2. **New +** → **Blueprint** → select **`Godliver143/c2d2-teambuilder-model`** (or your fork).
3. **`render.yaml`** provisions **`runtime: python`** with **no blueprint `rootDir`** (avoid path bugs with **`Combat Mission Model`** spaces — see [Render monorepo `rootDir`](https://render.com/docs/monorepo-support)). Build/start **`cd "Combat Mission Model"`** then **`python -m pip … -r requirements-docker.txt`** / **`python -m uvicorn …`**. **`PYTHON_VERSION=3.11.11`** and **`.python-version`** at the **repo root** keep wheels aligned with **`requirements-docker.txt`**.
4. **Apply** / sync the blueprint and wait for **build + deploy** (mostly `pip`; usually a few minutes).
5. Open the **`onrender.com`** URL from the dashboard.

**If you already had a Docker-based service from an older blueprint:** Render may keep the old runtime. Easiest fix: **create a new Web Service** from the updated blueprint, or manually match **`render.yaml`**: Language **Python 3**, **leave Root Directory empty** (repo root), and paste **`buildCommand`** / **`startCommand`** from that file verbatim.

---

**Option B — Web Service without Blueprint (manual, native Python)**

1. **New +** → **Web Service** → repo + branch **`main`**.
2. **Language:** **Python 3**.
3. **Root directory:** leave **empty** (repository root — same as **`render.yaml`**).
4. **Build command:** (copy from **`render.yaml`**)  
   `cd "Combat Mission Model" && python -m pip install --upgrade pip setuptools wheel && python -m pip install --no-cache-dir -r requirements-docker.txt`
5. **Start command:**  
   `cd "Combat Mission Model" && python -m uvicorn main:app --host 0.0.0.0 --port $PORT`
6. **Environment:** **`PYTHON_VERSION`** = **`3.11.11`**, **`CORS_ORIGINS`** = **`*`**.
7. **Health check path:** **`/health`**

---

**Option C — Docker (manual only)**

Useful for self-hosted Docker or hosts that prefer a container image; on Render Free, image builds sometimes fail (`Killed` / OOM).

1. **New +** → **Web Service** → repo **`main`**.
2. **Runtime:** **Docker**; **root directory:** **`Combat Mission Model`**; **Dockerfile:** **`Dockerfile`**
3. Add **`CORS_ORIGINS`** = **`*`**; **health check** **`/health`**

---

**After deploy — quick checks**

- Browser: **`https://YOUR_SERVICE.onrender.com/docs`** (Swagger)
- **`https://YOUR_SERVICE.onrender.com/ui`** (team UI — use **https** on Render)
- Postman base URL (example): **`https://YOUR_SERVICE.onrender.com`**  
  - `GET /health`  
  - `POST /team/select` with JSON body **as documented above** (no query params).

From repo root locally (after replacing the hostname):

```bash
python3 scripts/verify_live_api.py --base-url https://YOUR_SERVICE.onrender.com
```

Free tier **sleeps after idle**: first request after sleep can take **30–60+ seconds**.

**Security:** Treat the Render URL as **public**. Restrict later with SSO/VPN/API keys or a private service if needed.

---

**If the build fails on Render**

- Open **Logs** → **Build** (or **Deploy** if the build succeeded but the app exited). Note the **first** `ERROR` / `ModuleNotFoundError` / `Killed` line.
  - **`rootDir` + folder names with spaces** — **`render.yaml` no longer uses `rootDir`**; it clones the repo and runs **`cd "Combat Mission Model" && …`**. Sync the blueprint (or paste those commands manually) before debugging further.
  - **`Killed` during `pip install`** — try the **native Python** blueprint in **`render.yaml`** (not Docker); ensure **`PYTHON_VERSION`** is **`3.11.11`**. Avoid **`uvicorn[standard]`** on cloud builds (pinned **`requirements-docker.txt`** uses plain **`uvicorn`**).
  - **Wrong Python version** — new Render stacks default to a very new Python; this repo pins **3.11.11** in **`render.yaml`** and **`.python-version`** under **`Combat Mission Model/`**.
  - **Docker-only failures** (`COPY`, image build): use Option A/B (native Python) above, or a paid **Docker** pipeline with more memory.
- To refresh the model: **`python train.py`** locally → commit **`athena.joblib`** + **`models/mission_model_metadata.json`** → push → redeploy.

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
