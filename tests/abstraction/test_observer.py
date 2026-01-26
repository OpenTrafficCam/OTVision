from unittest.mock import AsyncMock, Mock

import pytest

from OTVision.abstraction.observer import (
    AsyncObservable,
    AsyncSubject,
    Observable,
    Subject,
)


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


class TestAsyncSubject:
    @pytest.fixture
    def target(self) -> AsyncSubject[int]:
        return AsyncSubject[int]()

    def test_register_observer(self, target: AsyncSubject[int]) -> None:
        given_observer = AsyncMock()
        target.register(given_observer)

        assert target._observers == [given_observer]

    @pytest.mark.asyncio
    async def test_notify_observers(self, target: AsyncSubject[int]) -> None:
        given_observer = AsyncMock()
        target.register(given_observer)
        await target.notify(1)

        given_observer.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_notify_multiple_observers_concurrently(
        self, target: AsyncSubject[int]
    ) -> None:
        observer1 = AsyncMock()
        observer2 = AsyncMock()
        observer3 = AsyncMock()

        target.register(observer1)
        target.register(observer2)
        target.register(observer3)

        await target.notify(42)

        observer1.assert_called_once_with(42)
        observer2.assert_called_once_with(42)
        observer3.assert_called_once_with(42)

    @pytest.mark.asyncio
    async def test_notify_with_no_observers(self, target: AsyncSubject[int]) -> None:
        # Should not raise an exception when no observers are registered
        await target.notify(1)

    def test_register_duplicate_observer(self, target: AsyncSubject[int]) -> None:
        given_observer = AsyncMock()
        target.register(given_observer)
        target.register(given_observer)

        # Should only have one instance of the observer
        assert len(target._observers) == 1


class IntAsyncObservable(AsyncObservable[int]):
    pass


class TestAsyncObservable:
    @pytest.fixture
    def subject(self) -> Mock:
        return Mock()

    @pytest.fixture
    def target(self, subject: AsyncSubject[int]) -> IntAsyncObservable:
        return IntAsyncObservable(subject)

    def test_register(self, target: IntAsyncObservable, subject: Mock) -> None:
        observer = AsyncMock()
        target.register(observer)

        subject.register.assert_called_once_with(observer)
