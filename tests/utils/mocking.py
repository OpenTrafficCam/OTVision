from unittest.mock import Mock


def create_mocks[T](amount: int) -> list[T]:
    return [Mock() for _ in range(amount)]
