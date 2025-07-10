from typing import Callable

StrIdGenerator = Callable[[], str]


class CurrentTrackingRunId:
    def __init__(self) -> None:
        self._current_tracking_run_id: str | None = None

    def update(self, tracking_run_id: str) -> None:
        self._current_tracking_run_id = tracking_run_id

    def get(self) -> str:
        if self._current_tracking_run_id is None:
            raise ValueError("Tracking run id is not set!")
        return self._current_tracking_run_id


class GenerateNewTrackingRunId:
    def __init__(
        self, id_generator: StrIdGenerator, current: CurrentTrackingRunId
    ) -> None:
        self._generate_id = id_generator
        self._current = current

    def generate(self) -> None:
        self._current.update(self._generate_id())


class GetCurrentTrackingRunId:
    def __init__(self, current: CurrentTrackingRunId) -> None:
        self._current = current

    def get(self) -> str:
        return self._current.get()
