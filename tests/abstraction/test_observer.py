import asyncio
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

        # Give background tasks time to complete
        await asyncio.sleep(0.1)

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

        # Give background tasks time to complete
        await asyncio.sleep(0.1)

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

    @pytest.mark.asyncio
    async def test_notify_with_observer_exception(
        self, target: AsyncSubject[int], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that exceptions in observers are caught and logged without
        affecting other observers.
        """
        successful_observer = AsyncMock()

        async def failing_observer(value: int) -> None:
            raise ValueError("Test exception from failing observer")

        another_successful_observer = AsyncMock()

        target.register(successful_observer)
        target.register(failing_observer)
        target.register(another_successful_observer)

        # notify() should not raise an exception
        await target.notify(42)

        # Give background tasks time to complete
        await asyncio.sleep(0.1)

        # Both successful observers should have been called
        successful_observer.assert_called_once_with(42)
        another_successful_observer.assert_called_once_with(42)

        # The exception should have been logged
        assert "Exception in async observer" in caplog.text
        assert "Test exception from failing observer" in caplog.text

    @pytest.mark.asyncio
    async def test_notify_returns_immediately(self, target: AsyncSubject[int]) -> None:
        """Test that notify() returns immediately without waiting for
        observers to complete.
        """
        slow_observer_started = asyncio.Event()
        slow_observer_finished = asyncio.Event()

        async def slow_observer(value: int) -> None:
            slow_observer_started.set()
            await asyncio.sleep(0.2)  # Simulate slow operation
            slow_observer_finished.set()

        target.register(slow_observer)

        # Call notify() - it should return immediately
        notify_start = asyncio.get_event_loop().time()
        await target.notify(42)
        notify_duration = asyncio.get_event_loop().time() - notify_start

        # notify() should return almost immediately (< 0.05s), not wait for
        # the slow observer
        assert notify_duration < 0.05

        # But the observer should still be running in the background
        await asyncio.wait_for(slow_observer_started.wait(), timeout=1.0)

        # Verify observer eventually completes
        await asyncio.wait_for(slow_observer_finished.wait(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_wait_for_all_observers(self, target: AsyncSubject[int]) -> None:
        """Test that wait_for_all_observers waits for all pending tasks."""
        observer_completed = asyncio.Event()

        async def slow_observer(value: int) -> None:
            await asyncio.sleep(0.1)
            observer_completed.set()

        target.register(slow_observer)
        await target.notify(42)

        # Observer should not have completed yet
        assert not observer_completed.is_set()

        # Wait for all observers to complete
        await target.wait_for_all_observers()

        # Now observer should have completed
        assert observer_completed.is_set()

    @pytest.mark.asyncio
    async def test_wait_for_all_observers_with_no_pending_tasks(
        self, target: AsyncSubject[int]
    ) -> None:
        """Test that wait_for_all_observers works with no pending tasks."""
        # Should not raise any exception
        await target.wait_for_all_observers()


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
