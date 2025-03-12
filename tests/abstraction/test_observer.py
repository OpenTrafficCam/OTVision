from unittest.mock import Mock

import pytest

from OTVision.abstraction.observer import Observable, Subject


class TestSubject:
    @pytest.fixture
    def target(self) -> Subject[int]:
        return Subject[int]()

    def test_register_observer(self, target: Subject[int]) -> None:
        given_observer = Mock()
        target.register(given_observer)

        assert target._observers == [given_observer]

    def test_notify_observers(self, target: Subject[int]) -> None:
        given_observer = Mock()
        target.register(given_observer)
        target.notify(1)

        given_observer.assert_called_once_with(1)


class IntObservable(Observable[int]):
    pass


class TestObservable:
    @pytest.fixture
    def subject(self) -> Mock:
        return Mock()

    @pytest.fixture
    def target(self, subject: Subject[int]) -> IntObservable:
        return IntObservable(subject)

    def test_register(self, target: IntObservable, subject: Mock) -> None:
        observer = Mock()
        target.register(observer)

        subject.register.assert_called_once_with(observer)
