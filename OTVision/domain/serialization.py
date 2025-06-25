from abc import ABC
from pathlib import Path


class Serializer(ABC):
    def serialize(self, file: Path) -> dict:
        raise NotImplementedError


class Deserializer(ABC):
    def deserialize(self, file: Path) -> dict:
        raise NotImplementedError
