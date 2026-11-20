"""Wafer map basics shared by the ETL, the models and the API.

A WM-811K wafer map is a 2D grid of die codes:
  0 = no die (outside the round wafer)
  1 = die passed the electrical test
  2 = die failed
"""
from __future__ import annotations

import numpy as np

OFF, PASS, FAIL = 0, 1, 2
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
SIZE = 64  # every map is resized to SIZE x SIZE for the models


def resize(wafer: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Nearest-neighbour resize. Keeps the codes as 0, 1, 2 (interpolation would invent 1.5)."""
    rows = (np.arange(size) * wafer.shape[0] / size).astype(int)
    cols = (np.arange(size) * wafer.shape[1] / size).astype(int)
    return wafer[np.ix_(rows, cols)]


def pool(wafer: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Resize that never loses a fail: each output cell takes the max code of the cells it covers.

    Codes are ordered OFF < PASS < FAIL, so the max is FAIL if any covered die failed. When a map
    is shrunk (2% of WM-811K maps are over 64 on a side), nearest-neighbour sampling skips rows and
    columns and can drop a one-die-wide scratch; this keeps it. When a map is enlarged it is the same
    as nearest-neighbour (reduceat returns the single element for repeated indices)."""
    rows = (np.arange(size) * wafer.shape[0] / size).astype(int)
    cols = (np.arange(size) * wafer.shape[1] / size).astype(int)
    return np.maximum.reduceat(np.maximum.reduceat(wafer, rows, axis=0), cols, axis=1)


def to_channels(wafer: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Two float channels for the CNN: where the wafer is, and where it failed."""
    m = pool(wafer, size)
    return np.stack([(m > OFF), (m == FAIL)]).astype(np.float32)


def die_yield(wafer: np.ndarray) -> float:
    dies = int((wafer > OFF).sum())
    return float((wafer == PASS).sum() / dies) if dies else float("nan")
