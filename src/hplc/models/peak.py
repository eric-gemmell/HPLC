import copy


class Peak():
    def __init__(
        self,
        retention_time: float,
        area: float,
        start_time: float,
        end_time: float,
        unit: str,
        compound: str | None = None,
    ) -> None:
        self.retention_time = retention_time
        self.area = area
        self.start_time = start_time
        self.end_time = end_time
        self.unit = unit
        self.compound = compound

    def _copy(self) -> Peak:
        return copy.deepcopy(self)


class Peak2D(Peak):
    def __init__(
        self,
        retention_time: float,
        area: float,
        start_time: float,
        end_time: float,
        max_height: float,
        unit: str,
        mu: float | None = None,
        amplitude: float | None = None,
        sigma: float | None = None,
        alpha: float | None = None,
        compound: str | None = None,
    ) -> None:
        super().__init__(
            retention_time=retention_time,
            area=area,
            start_time=start_time,
            end_time=end_time,
            unit=unit,
            compound=compound,
        )
        self.max_height = max_height
        self.mu = mu
        self.amplitude = amplitude
        self.sigma = sigma
        self.alpha = alpha

    def __repr__(self) -> str:
        return (
            f"Peak2D(retention_time={self.retention_time}, "
            f"max_height={self.max_height}, "
            f"area={self.area}, "
            f"start_time={self.start_time}, "
            f"end_time={self.end_time}, "
            f"unit='{self.unit}')"
        )