#!/usr/bin/env python3
"""Verify a running API (localhost or deployed) is reachable and team-facing.

Usage (from repo root):
  python3 scripts/verify_live_api.py --base-url https://your-service.onrender.com

Or:
  BASE_URL=https://your-api.example.com python3 scripts/verify_live_api.py
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request


def _get(base: str, path: str) -> tuple[int, str]:
    req = urllib.request.Request(f"{base}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode("utf-8", errors="replace")
            return r.status, body[:2000]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def _options(base: str, path: str, origin: str) -> tuple[int, dict[str, str]]:
    req = urllib.request.Request(
        f"{base}{path}",
        method="OPTIONS",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            return r.status, hdrs
    except urllib.error.HTTPError as e:
        hdrs = {k.lower(): v for k, v in e.headers.items()} if e.headers else {}
        return e.code, hdrs


def _post_json(base: str, path: str, payload: bytes) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{base}{path}",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode("utf-8", errors="replace")
            return r.status, body[:1200]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def main() -> int:
    default = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    parser = argparse.ArgumentParser(description="Check mission API responsiveness (HTTP).")
    parser.add_argument(
        "--base-url",
        default=default,
        help="Public base URL without trailing slash",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"Checking {base}")
    errs = 0

    code, body = _get(base, "/health")
    if code != 200:
        errs += 1
        print(f"  FAIL GET /health -> {code}\n    {body[:500]}")
    else:
        print(f"  OK   GET /health -> {code} ({len(body)} chars)")
        if '"status"' not in body or "ok" not in body.lower():
            print("  WARN /health payload missing expected status text")
        if '"model_trained": false' in body.replace(" ", ""):
            errs += 1
            print("  FAIL model appears untrained")

    code2, _ = _get(base, "/openapi.json")
    print(f"  {'OK ' if code2 == 200 else 'FAIL'} GET /openapi.json -> {code2}")
    if code2 != 200:
        errs += 1

    payload = b'{"mission_type":"ambush","top_k":20,"num_team_options":2}'
    code3, body3 = _post_json(base, "/team/select", payload)
    if code3 != 200:
        errs += 1
        print(f"  FAIL POST /team/select -> {code3}\n    {body3[:400]}")
    else:
        print(f"  OK   POST /team/select -> {code3} ({len(body3)} chars prefix)")

    for origin in ("https://partner-app.invalid",):
        c, hdr = _options(base, "/team/select", origin)
        allow = hdr.get("access-control-allow-origin", "")
        star_ok = allow == "*"
        mirror_ok = allow == origin
        ok = c in (200, 204) and (star_ok or mirror_ok)
        print(
            f"  {'OK ' if ok else 'WARN'} OPTIONS /team/select Origin={origin!r} -> {c}; "
            f"allow-origin={allow!r}"
        )
        if c not in (200, 204):
            errs += 1
        if not allow:
            print("       Browsers need Access-Control-Allow-Origin for cross-site POSTs.")

    if errs:
        print(f"\nCompleted with {errs} hard failure(s).")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
