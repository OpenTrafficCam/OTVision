from typing import Callable, TypeVar

VALUE = TypeVar("VALUE")
type Observer[T] = Callable[[T], None]


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
