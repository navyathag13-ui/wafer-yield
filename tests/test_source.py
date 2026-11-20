import io
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from wafer_yield.source import SafeUnpickler, records


class Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_blocks_code_execution():
    with pytest.raises(pickle.UnpicklingError, match="blocked"):
        SafeUnpickler(io.BytesIO(pickle.dumps(Evil()))).load()


def test_blocks_pandas_readers():
    class Reader:
        def __reduce__(self):
            return (pd.read_csv, ("http://example.com/x.csv",))
    with pytest.raises(pickle.UnpicklingError):
        SafeUnpickler(io.BytesIO(pickle.dumps(Reader()))).load()


def test_reads_a_wm811k_shaped_frame(tmp_path):
    df = pd.DataFrame({
        "waferMap": [np.array([[0, 1], [2, 1]], dtype=np.uint8), np.array([[1]], dtype=np.uint8)],
        "dieSize": [3.0, 1.0], "lotName": ["lot1", "lot1"], "waferIndex": [1.0, 2.0],
        "trianTestLabel": [np.array([["Training"]]), np.zeros((0, 0))],
        "failureType": [np.array([["Loc"]]), np.zeros((0, 0))],
    })
    path = tmp_path / "mini.pkl"
    df.to_pickle(path)
    rows = list(records(path))
    assert len(rows) == 2 and rows[0]["lotName"] == "lot1"
    assert np.array_equal(rows[0]["waferMap"], [[0, 1], [2, 1]])
