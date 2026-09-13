from __future__ import annotations

import numpy as np
from pybaselines import Baseline


def compute_baseline(
    time: np.ndarray,
    signal: np.ndarray,
    **kwargs,
) -> np.ndarray:

    signal = np.asarray(signal, dtype=float)
    if len(time) != len(signal):
        raise ValueError(
            f"time and signal must have the same length, "
            f"got {len(time)} and {len(signal)}"
        )

    baseline, _ = Baseline(x_data=time).snip(signal, **kwargs)
    return baseline
