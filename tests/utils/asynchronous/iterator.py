from typing import AsyncIterator

from OTVision.domain.frame import Frame


async def get_elements_of[T](iterator: AsyncIterator[T]) -> list[T]:
    elements = []
    async for elem in iterator:
        elements.append(elem)
    return elements


async def async_frame_generator(frames: list[Frame]) -> AsyncIterator[Frame]:
    for frame in frames:
        yield frame
