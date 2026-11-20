"""Raw WM-811K records -> checked Parquet tables.

Every wafer is either loaded (maybe with soft flags) or quarantined with a reason, so the
counts always add up to the input:
  data/wafers.parquet      one row per usable wafer: lot, index, split, label, yield, map bytes
  data/quarantine.parquet  rows that could not be used, with why

Hard problems (quarantine): the map is not a 2D grid, holds codes other than 0/1/2, or has no dies.
Soft problems (flag, keep): the stored dieSize disagrees with the dies on the map, the wafer
index is outside 1-25 (a lot holds 25 wafers), or the same lot and index appear twice.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .maps import CLASSES, FAIL, OFF, PASS

WAFER_SCHEMA = pa.schema([
    ("lot", pa.string()), ("wafer_index", pa.int16()), ("split", pa.string()), ("label", pa.string()),
    ("rows", pa.int16()), ("cols", pa.int16()), ("dies", pa.int32()), ("fails", pa.int32()),
    ("yield", pa.float32()), ("flags", pa.string()), ("map", pa.binary()),
])
QUARANTINE_SCHEMA = pa.schema([
    ("source_row", pa.int64()), ("lot", pa.string()), ("wafer_index", pa.int16()), ("reason", pa.string()),
])


def unwrap(value) -> str | None:
    """Labels are stored as nested arrays like [['Training']]; unlabeled ones are empty arrays."""
    arr = np.asarray(value, dtype=object).ravel()
    if arr.size == 0:
        return None
    text = str(arr[0]).strip()
    return text or None


def check(wafer) -> tuple[np.ndarray | None, str | None]:
    """Return the map as uint8, or None and the reason it is unusable."""
    arr = np.asarray(wafer)
    if arr.ndim != 2 or arr.size == 0:
        return None, f"not a 2D map (shape {arr.shape})"
    codes = np.unique(arr)
    if not np.isin(codes, (OFF, PASS, FAIL)).all():
        return None, f"unknown die codes {sorted(set(codes.tolist()) - {OFF, PASS, FAIL})[:5]}"
    arr = arr.astype(np.uint8)
    if not (arr > OFF).any():
        return None, "no dies on the map"
    return arr, None


def process(records: Iterable[dict], out_dir: Path, batch: int = 20000) -> dict:
    """Stream records (keys: waferMap, dieSize, lotName, waferIndex, trianTestLabel, failureType)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    seen: set[tuple[str, int]] = set()
    rows: list[dict] = []
    bad: list[dict] = []
    writer = pq.ParquetWriter(out_dir / "wafers.parquet", WAFER_SCHEMA, compression="zstd")

    def flush():
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=WAFER_SCHEMA))
            rows.clear()

    for i, rec in enumerate(records):
        stats["input"] += 1
        lot = str(rec.get("lotName", "")).strip() or None
        idx_raw = rec.get("waferIndex")
        idx = int(idx_raw) if idx_raw is not None and np.isfinite(idx_raw) else -1
        wafer, reason = check(rec.get("waferMap"))
        if wafer is None:
            bad.append({"source_row": i, "lot": lot, "wafer_index": idx, "reason": reason})
            stats["quarantined"] += 1
            continue
        label = unwrap(rec.get("failureType"))
        if label is not None and label not in CLASSES:
            bad.append({"source_row": i, "lot": lot, "wafer_index": idx, "reason": f"unknown label {label!r}"})
            stats["quarantined"] += 1
            continue
        dies, fails = int((wafer > OFF).sum()), int((wafer == FAIL).sum())
        flags = []
        die_size = rec.get("dieSize")
        if die_size is not None and np.isfinite(die_size) and int(die_size) != dies:
            flags.append("die_count_mismatch")
        if not 1 <= idx <= 25:
            flags.append("wafer_index_out_of_range")
        if (lot, idx) in seen:
            flags.append("duplicate_lot_index")
        seen.add((lot, idx))
        for f in flags:
            stats[f"flag:{f}"] += 1
        stats["loaded"] += 1
        stats["labeled" if label else "unlabeled"] += 1
        rows.append({
            "lot": lot, "wafer_index": idx, "split": unwrap(rec.get("trianTestLabel")), "label": label,
            "rows": wafer.shape[0], "cols": wafer.shape[1], "dies": dies, "fails": fails,
            "yield": (dies - fails) / dies, "flags": ",".join(flags), "map": wafer.tobytes(),
        })
        if len(rows) >= batch:
            flush()
    flush()
    writer.close()
    pq.write_table(pa.Table.from_pylist(bad, schema=QUARANTINE_SCHEMA), out_dir / "quarantine.parquet")
    assert stats["loaded"] + stats["quarantined"] == stats["input"]
    return dict(stats)


def decode(row) -> np.ndarray:
    """Rebuild a map from its Parquet row (map bytes plus rows and cols)."""
    return np.frombuffer(row["map"], dtype=np.uint8).reshape(int(row["rows"]), int(row["cols"]))
