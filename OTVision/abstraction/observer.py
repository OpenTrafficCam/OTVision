import asyncio
from typing import Awaitable, Callable, TypeVar

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
    The order that registered observers are notified is dictated by the order they have
    been registered. Meaning, first to be registered is first to be notified.
    All observers are awaited concurrently using asyncio.gather().
    """

    def __init__(self) -> None:
        self._observers: list[AsyncObserver[T]] = []

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

        All observers are notified concurrently using asyncio.gather().

        Args:
            value (T): value to notify the observer with.
        """
        if self._observers:
            await asyncio.gather(*[observer(value) for observer in self._observers])


class AsyncObservable[T]:
    def __init__(self, subject: AsyncSubject[T]) -> None:
        self._subject = subject

    def register(self, observer: AsyncObserver[T]) -> None:
        self._subject.register(observer)
