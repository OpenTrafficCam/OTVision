from typing import Generator


def make_generator[T](_list: list[T]) -> Generator[T, None, None]:
    yield from _list
