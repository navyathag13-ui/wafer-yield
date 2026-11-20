"""Read LSWMD.pkl (the WM-811K release) without trusting it.

The dataset ships as a pandas pickle. A plain pickle.load runs whatever callables the file
names, so a tampered copy could run code on this machine. This loader only lets the pickle
build numpy arrays and the pandas classes a DataFrame is made of; anything else
(os.system, subprocess, eval, pandas readers) raises before it is called.

The file is from an old pandas, so some class paths moved (pandas.indexes -> pandas.core.indexes).
"""
from __future__ import annotations

import _compat_pickle
import pickle
from pathlib import Path
from typing import Iterator

from pandas.compat.pickle_compat import Unpickler as PandasUnpickler  # also remaps renamed pandas classes

# Exact names a DataFrame of numpy arrays needs, old pandas and new.
ALLOWED = {
    ("builtins", n) for n in ("object", "slice", "set", "frozenset", "list", "dict", "tuple",
                              "complex", "bytearray", "range", "int", "float", "str", "bool")
} | {
    ("copyreg", "_reconstructor"), ("collections", "OrderedDict"), ("_codecs", "encode"),
    ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy", "bool_"),
    ("numpy._core.numeric", "_frombuffer"), ("numpy.core.numeric", "_frombuffer"),
    ("pandas", "DataFrame"), ("pandas", "Series"), ("pandas", "Index"), ("pandas", "RangeIndex"),
    ("pandas", "StringDtype"), ("pandas.arrays", "ArrowStringArray"),
    ("pyarrow.lib", "_restore_array"), ("pyarrow.lib", "type_for_alias"), ("pyarrow.lib", "py_buffer"),
}
# Whole modules that only hold data containers and their rebuild helpers.
ALLOWED_PREFIXES = ("numpy.core.multiarray", "numpy._core.multiarray", "pandas.core.indexes.",
                    "pandas.core.internals", "pandas._libs.internals", "pandas._libs.lib",
                    "pandas.core.frame", "pandas.core.series", "pandas.core.dtypes.", "pandas.core.arrays.")


class SafeUnpickler(PandasUnpickler):
    def find_class(self, module: str, name: str):
        # The release is a Python 2 pickle: translate __builtin__, copy_reg and so on first,
        # so the allowlist below is checked against the real Python 3 names.
        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[(module, name)]
        elif module in _compat_pickle.IMPORT_MAPPING:
            module = _compat_pickle.IMPORT_MAPPING[module]
        if module == "pandas.core.internals":
            module = "pandas.core.internals.managers"
        elif module.startswith("pandas.indexes."):
            module = "pandas.core.indexes." + module[len("pandas.indexes."):]
        risky = name.lower().startswith(("read", "eval", "query", "to_", "_eval", "exec", "system"))
        if (module, name) in ALLOWED or (module.startswith(ALLOWED_PREFIXES) and not risky):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}: not needed for a DataFrame of arrays")


def load_frame(path: Path):
    with open(path, "rb") as fh:
        try:
            return SafeUnpickler(fh).load()
        except UnicodeDecodeError:  # a Python 2 pickle stores numpy buffers as str
            fh.seek(0)
            return SafeUnpickler(fh, encoding="latin1").load()


FIELDS = ("waferMap", "dieSize", "lotName", "waferIndex", "trianTestLabel", "failureType")


def records(path: Path) -> Iterator[dict]:
    df = load_frame(path)
    missing = set(FIELDS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    cols = [df[c].to_numpy() for c in FIELDS]
    for row in zip(*cols):
        yield dict(zip(FIELDS, row))
