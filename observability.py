import asyncio
import logging
import time
from functools import wraps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("travelstack")


def trace(fn):
    if asyncio.iscoroutinefunction(fn):
        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info(f"[START] {fn.__qualname__}")
            try:
                result = await fn(*args, **kwargs)
                logger.info(f"[END]   {fn.__qualname__} ({time.perf_counter() - start:.3f}s)")
                return result
            except Exception as e:
                logger.error(f"[ERROR] {fn.__qualname__}: {e}")
                raise
        return async_wrapper

    @wraps(fn)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        logger.info(f"[START] {fn.__qualname__}")
        try:
            result = fn(*args, **kwargs)
            logger.info(f"[END]   {fn.__qualname__} ({time.perf_counter() - start:.3f}s)")
            return result
        except Exception as e:
            logger.error(f"[ERROR] {fn.__qualname__}: {e}")
            raise
    return sync_wrapper
