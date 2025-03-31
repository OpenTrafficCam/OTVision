import uuid
from typing import Callable, Iterator

StrIdGenerator = Callable[[], str]


def tracking_run_uuid_generator() -> str:
    return str(uuid.uuid4())


def track_id_generator() -> Iterator[int]:
    track_id: int = 0
    while True:
        track_id += 1
        yield track_id
