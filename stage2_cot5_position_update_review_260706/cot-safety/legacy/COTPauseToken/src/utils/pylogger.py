import logging
from typing import Mapping, Optional


class RankedLogger(logging.LoggerAdapter):
    """Small logger adapter kept for compatibility with the original entrypoint."""

    def __init__(self, name: str = __name__, extra: Optional[Mapping[str, object]] = None) -> None:
        super().__init__(logger=logging.getLogger(name), extra=extra or {})
