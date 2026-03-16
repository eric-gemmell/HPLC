from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Callable

from hplc.core.signal import Signal


class Chromatogram:

    def __init__(
        self,
        signals: dict[str, Signal],
        filename: str
        additional_data: dict | None = None,
    ):
        if not signals:
            raise ValueError("Chromatogram must contain at least one signal")

        self.signals: dict[str, Signal] = signals
        self.filename: str = filename
        self.additional_data: dict = additional_data or {}
        self.peaks: list = []


    def _copy(self) -> Chromatogram:
        return copy.deepcopy(self)

    def display(self) -> Chromatogram:
        raise NotImplementedError(
            "Use Signal2D from the package __init__ which has display wired in"
        )