from abc import ABC, abstractmethod
from typing import Generator


class Filter[IN, OUT](ABC):
    """Abstract base class for implementing pipe-and-filter pattern filters.

    A filter processes a stream of input elements and produces a stream of output
    elements. This class defines the interface for all concrete filter implementations.

    Type Parameters:
        IN: The type of elements that the filter receives as input.
        OUT: The type of elements that the filter produces as output.
    """

    @abstractmethod
    def filter(self, pipe: Generator[IN, None, None]) -> Generator[OUT, None, None]:
        """Process elements from the input pipe and produce output elements.

        Args:
            pipe (Generator[IN, None, None]): Input stream of elements to be processed.

        Returns:
            Generator[OUT, None, None]: Output stream of processed elements.

        """

        raise NotImplementedError
