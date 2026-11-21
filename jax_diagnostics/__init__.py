"""Attach repair guides to exceptions at a Python application boundary."""

from collections.abc import Iterator
from contextlib import contextmanager
import traceback

from ._routing import Diagnosis, diagnose

__all__ = ["Diagnosis", "annotate_errors", "diagnose"]

DEFAULT_GUIDE_BASE = "https://github.com/mlnomadpy/jax-diagnostics/blob/main"


@contextmanager
def annotate_errors(*, guide_base: str = DEFAULT_GUIDE_BASE) -> Iterator[None]:
    """Add a repair note to a matching exception and re-raise it unchanged.

    Pass a mirror or a pinned revision as guide_base. Python 3.11+ displays
    exception notes in its standard traceback.
    """
    try:
        yield
    except Exception as error:
        try:
            source = "".join(traceback.format_exception(error))
            route = diagnose(source)
            if route is not None:
                link = f"{guide_base.rstrip('/')}/{route.link}"
                note = f"[{route.id}] {route.title}\nAgent repair guide: {link}"
                if note not in getattr(error, "__notes__", ()):
                    error.add_note(note)
        except Exception:
            # Diagnostics must never obscure the original application failure.
            pass
        raise
