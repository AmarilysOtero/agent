from __future__ import annotations
import logging, sys

def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    if not root.handlers:
        handler.setFormatter(fmt)
        root.addHandler(handler)
    root.setLevel(level)
