import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

from OTVision.helpers.log import LOGGER_NAME

VALUE = TypeVar("VALUE")
type Observer[T] = Callable[[T], None]
type AsyncObserver[T] = Callable[[T], Awaitable[None]]


class Subject[T]:
    """Generic subject class to handle and notify observers.

    This class ensures that no duplicate observers can be registered.
    The order that registered observers are notified is dictated by the order they have
    been registered. Meaning, first to be registered is first to be notified.
    """

    def __init__(self) -> None:
        self._observers: list[Observer[T]] = []

    def register(self, observer: Observer[T]) -> None:
        """Listen to changes of subject.

        Args:
            observer (OBSERVER): the observer to be registered. This must be a
                `Callable`.
        """
        new_observers = self._observers.copy()
        new_observers.append(observer)
        self._observers = list(dict.fromkeys(new_observers))

    def notify(self, value: T) -> None:
        """Notifies observers about the list of tracks.

        Args:
            value (VALUE): value to notify the observer with.
        """
        [notify_observer(value) for notify_observer in self._observers]


class Observable[T]:
    def __init__(self, subject: Subject[T]) -> None:
        self._subject = subject

    def register(self, observer: Observer[T]) -> None:
        self._subject.register(observer)


class AsyncSubject[T]:
    """Generic async subject class to handle and notify async observers.

    This class ensures that no duplicate observers can be registered.
    The order that registered observers are notified is dictated by the order
    they have been registered. Meaning, first to be registered is first to be
    notified. Observers are executed as fire-and-forget background tasks for
    non-blocking notifications.
    """

    def __init__(self) -> None:
        self._observers: list[AsyncObserver[T]] = []
        self._pending_tasks: set[asyncio.Task[None]] = set()

    def register(self, observer: AsyncObserver[T]) -> None:
        """Listen to changes of subject.

        Args:
            observer (AsyncObserver[T]): the observer to be registered. This must be an
                async `Callable` that returns an `Awaitable`.
        """
        new_observers = self._observers.copy()
        new_observers.append(observer)
        self._observers = list(dict.fromkeys(new_observers))

    async def notify(self, value: T) -> None:
        """Notifies observers about the value asynchronously.

        All observers are notified as fire-and-forget background tasks,
        allowing the event loop to continue processing while observers execute.
        Each observer runs independently; exceptions are caught and logged
        without affecting other observers or blocking the caller.

        Args:
            value (T): value to notify the observer with.
        """
        log = logging.getLogger(LOGGER_NAME)

        async def _safe_observer_call(observer: AsyncObserver[T], value: T) -> None:
            """Wrapper to safely call an observer and log any exceptions."""
            try:
                await observer(value)
            except Exception as e:
                observer_name = (
                    observer.__name__
                    if hasattr(observer, "__name__")
                    else repr(observer)
                )
                log.error(
                    f"Exception in async observer {observer_name}: {e}",
                    exc_info=True,
                )

        for observer in self._observers:
            task = asyncio.create_task(_safe_observer_call(observer, value))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

    async def wait_for_all_observers(self) -> None:
        """Wait for all pending observer tasks to complete.

        This method can be used when you need to ensure all observers have
        finished processing before proceeding. Useful in tests or when
        synchronization is required.
        """
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)


class AsyncObservable[T]:
    def __init__(self, subject: AsyncSubject[T]) -> None:
        self._subject = subject

    def register(self, observer: AsyncObserver[T]) -> None:
        self._subject.register(observer)

    async def wait_for_all_observers(self) -> None:
        """Wait for all pending observer tasks to complete.

        This method delegates to the subject's wait_for_all_observers method.
        """
        await self._subject.wait_for_all_observers()
