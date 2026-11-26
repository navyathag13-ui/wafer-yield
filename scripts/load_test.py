"""Load test for the wafer API: step up concurrency, report latency percentiles, throughput and errors.

Traffic mix per request: 70% POST /classify, 20% POST /similar, 10% GET /spc/signals, using synthetic
wafer maps (40x40, eight pattern types) so no dataset is needed.

  python scripts/load_test.py --url http://localhost:8010 --label local --levels 1 5 10 20 --seconds 30
Results go to results/load_test_<label>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.synthetic import make  # noqa: E402

KINDS = ["Center", "Edge-Ring", "Donut", "Scratch", "Loc", "Edge-Loc", "Random", "none"]


def maps() -> list[list[list[int]]]:
    out = []
    for i, k in enumerate(KINDS * 3):
        try:
            out.append(make(k, n=40, seed=i).tolist())
        except Exception:
            continue
    return out


async def worker(client, stop, wafers, lat, errors, codes):
    while time.perf_counter() < stop:
        r = random.random()
        t0 = time.perf_counter()
        try:
            if r < 0.7:
                resp = await client.post("/classify", json={"wafer_map": random.choice(wafers)})
            elif r < 0.9:
                resp = await client.post("/similar", json={"wafer_map": random.choice(wafers), "k": 10})
            else:
                resp = await client.get("/spc/signals", params={"limit": 20})
            codes[resp.status_code] = codes.get(resp.status_code, 0) + 1
            if resp.status_code == 200:
                lat.append((time.perf_counter() - t0) * 1000)
            else:
                errors.append(resp.status_code)
        except httpx.HTTPError as e:
            errors.append(type(e).__name__)
            codes[type(e).__name__] = codes.get(type(e).__name__, 0) + 1


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))], 1) if xs else None


async def level(url, n, seconds, wafers):
    lat, errors, codes = [], [], {}
    limits = httpx.Limits(max_connections=n, max_keepalive_connections=n)
    async with httpx.AsyncClient(base_url=url, timeout=60, limits=limits) as client:
        t0 = time.perf_counter()
        stop = t0 + seconds
        await asyncio.gather(*(worker(client, stop, wafers, lat, errors, codes) for _ in range(n)))
        elapsed = time.perf_counter() - t0
    total = len(lat) + len(errors)
    return {"concurrency": n, "seconds": round(elapsed, 1), "requests": total, "ok": len(lat),
            "errors": len(errors), "error_rate": round(len(errors) / total, 4) if total else None,
            "rps": round(len(lat) / elapsed, 1), "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95),
            "p99_ms": pct(lat, 99), "mean_ms": round(statistics.mean(lat), 1) if lat else None,
            "status_codes": {str(k): v for k, v in codes.items()}}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--seconds", type=int, default=30)
    a = ap.parse_args()
    wafers = maps()
    async with httpx.AsyncClient(base_url=a.url, timeout=120) as c:   # warm up (cold start, model load)
        t0 = time.perf_counter()
        (await c.get("/health")).raise_for_status()
        warm_ms = round((time.perf_counter() - t0) * 1000)
    rows = []
    for n in a.levels:
        row = await level(a.url, n, a.seconds, wafers)
        rows.append(row)
        print(json.dumps(row))
    out = {"label": a.label, "url_host": httpx.URL(a.url).host, "first_health_ms": warm_ms,
           "mix": "70% classify, 20% similar, 10% spc/signals", "levels": rows,
           "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    path = Path(__file__).resolve().parents[1] / "results" / f"load_test_{a.label}.json"
    path.write_text(json.dumps(out, indent=2))
    print("saved", path)


if __name__ == "__main__":
    asyncio.run(main())
