from typing import AsyncIterator


async def get_elements_of[T](iterator: AsyncIterator[T]) -> list[T]:
    elements = []
    async for elem in iterator:
        elements.append(elem)
    return elements
