from abc import ABC, abstractmethod
from datetime import datetime, timezone


class DatetimeProvider(ABC):
    @abstractmethod
    def provide(self) -> datetime:
        raise NotImplementedError


class CurrentDatetimeProvider(DatetimeProvider):
    def provide(self) -> datetime:
        return datetime.now(tz=timezone.utc)
