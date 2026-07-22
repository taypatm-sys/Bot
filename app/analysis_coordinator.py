import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AnalysisCoordinator:
    """Serializes Gemini analysis and gives interactive requests priority."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active = False
        self._interactive_waiters = 0

    @asynccontextmanager
    async def interactive(self) -> AsyncIterator[None]:
        acquired = False
        async with self._condition:
            self._interactive_waiters += 1
            try:
                await self._condition.wait_for(lambda: not self._active)
                self._active = True
                acquired = True
            finally:
                self._interactive_waiters -= 1
                self._condition.notify_all()
        try:
            yield
        finally:
            if acquired:
                async with self._condition:
                    self._active = False
                    self._condition.notify_all()

    @asynccontextmanager
    async def background(self) -> AsyncIterator[None]:
        acquired = False
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._active and self._interactive_waiters == 0
            )
            self._active = True
            acquired = True
        try:
            yield
        finally:
            if acquired:
                async with self._condition:
                    self._active = False
                    self._condition.notify_all()
