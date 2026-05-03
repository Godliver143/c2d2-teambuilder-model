"""
Smoke tests for main:app — run::

    cd "Combat Mission Model" && python3 -m unittest discover -v -s tests
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


class TestSmokeAPI(unittest.TestCase):
    """Hit core routes; TestClient triggers lifespan startup (loads joblib artifact)."""

    @classmethod
    def setUpClass(cls):
        os.chdir(ROOT)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        os.environ.setdefault("CORS_ORIGINS", "*")

        import main as main_mod

        cls.client_ctx = TestClient(main_mod.app, raise_server_exceptions=True)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("service"), "combat-mission-model")

    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data.get("model_trained"))
        self.assertGreater(data.get("soldiers_loaded", 0), 0)

    def test_openapi_docs(self):
        r = self.client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        self.assertIn("openapi", r.json())

    def test_metadata(self):
        r = self.client.get("/model/metadata")
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertGreaterEqual(payload.get("schema_version", 0), 1)
        mt = (payload.get("model") or {}).get("mission_type_classes") or []
        self.assertTrue(mt, msg="model.mission_type_classes should be present")

    def test_mission_types(self):
        r = self.client.get("/mission-types")
        self.assertEqual(r.status_code, 200)
        self.assertIn("ambush", r.json()["available_mission_types"])

    def test_team_select(self):
        r = self.client.post(
            "/team/select",
            json={
                "mission_type": "ambush",
                "top_k": 20,
                "num_team_options": 2,
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(len(data["team"]), 6)
        self.assertGreater(len(data["team_options"]), 0)

    def test_rankings_path(self):
        r = self.client.get("/soldiers/rankings/ambush")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.json().get("rankings", [])), 0)

    def test_soldiers_list_and_one_profile(self):
        r = self.client.get("/soldiers")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertGreater(body.get("total", 0), 0)
        sid = body["soldiers"][0]["leader_identifier"]
        r2 = self.client.get(f"/soldiers/{sid}")
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["leader_identifier"], sid)

    def test_team_select_bad_mission_type(self):
        r = self.client.post(
            "/team/select",
            json={"mission_type": "invalid_mission_xyz", "top_k": 20},
        )
        self.assertEqual(r.status_code, 400)

    def test_cors_headers_on_get(self):
        r = self.client.get(
            "/health",
            headers={"Origin": "https://example-teammate.com"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("access-control-allow-origin"), "*")

    def test_cors_preflight_options(self):
        r = self.client.options(
            "/team/select",
            headers={
                "Origin": "https://example-teammate.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertIn(r.status_code, (200, 204))
        self.assertEqual(r.headers.get("access-control-allow-origin"), "*")
        max_age = r.headers.get("access-control-max-age")
        self.assertIsNotNone(max_age, "preflight should set access-control-max-age")
        self.assertGreater(int(max_age), 0)
