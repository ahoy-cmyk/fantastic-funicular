"""Bridge for handling asyncio operations across different event loops."""

import asyncio
import concurrent.futures
from typing import Any, Coroutine, TypeVar

T = TypeVar('T')


class AsyncBridge:
    """Handles async operations across different event loops.
    
    This is needed when GUI frameworks like Kivy run on one event loop
    while subprocess operations run on another.
    """
    
    def __init__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._loop = None
    
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop to use for operations."""
        self._loop = loop
    
    async def run_in_loop(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine in the correct event loop.
        
        If we're already in the right loop, just await it.
        Otherwise, schedule it in the correct loop.
        """
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        
        # If we're in the same loop or no specific loop is set, just run it
        if current_loop == self._loop or self._loop is None:
            return await coro
        
        # Otherwise, run in the specified loop
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return await asyncio.wrap_future(future)
    
    def cleanup(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False)