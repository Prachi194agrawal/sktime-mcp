import asyncio
import threading
from concurrent.futures import Future

_bg_loop = None
_bg_thread = None
_lock = threading.Lock()


def _run_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def get_background_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop, _bg_thread

    with _lock:
        if _bg_loop is None or _bg_loop.is_closed():
            _bg_loop = asyncio.new_event_loop()
            _bg_thread = threading.Thread(
                target=_run_loop,
                args=(_bg_loop,),
                name="sktime-mcp-async-worker",
                daemon=True,
            )
            _bg_thread.start()

    return _bg_loop


def submit_coroutine(coro) -> Future:
    loop = get_background_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop)