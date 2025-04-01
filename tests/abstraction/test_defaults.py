import pytest

from OTVision.abstraction.defaults import value_or_default


class TestValueOrDefault:
    @pytest.mark.parametrize(
        "value, default, expected",
        [
            (5, 10, 5),
            ("hello", "default", "hello"),
            ([1, 2, 3], [4, 5, 6], [1, 2, 3]),
        ],
    )
    def test_returns_value_when_not_none[
        T
    ](self, value: T, default: T, expected: T) -> None:
        assert value_or_default(value, default) == expected

    @pytest.mark.parametrize(
        "value, default, expected",
        [
            (None, 10, 10),
            (None, "default", "default"),
            (None, [4, 5, 6], [4, 5, 6]),
        ],
    )
    def test_returns_default_when_value_is_none[
        T
    ](self, value: None, default: T, expected: T) -> None:
        assert value_or_default(value, default) == expected

    @pytest.mark.parametrize(
        "value, default, expected",
        [
            (None, 3.14, 3.14),
            ("test", "default", "test"),
        ],
    )
    def test_mixed_types[T](self, value: T | None, default: T, expected: T) -> None:
        assert value_or_default(value, default) == expected
