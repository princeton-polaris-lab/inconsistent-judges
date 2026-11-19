import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: The file ``ensure_job_logging`` opens in a job's output directory.
LOG_FNAME = "trainer.log"

def log_with_header(content: str) -> None:
    logger.info("=" * 60)
    logger.info(content)
    logger.info("=" * 60)

def ensure_job_logging(out_dir: Path) -> None:
    """Re-attach the console and file handlers when a library has reset the root logger."""
    root = logging.getLogger()

    # A FileHandler is a StreamHandler too, but it writes no console.
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                      for h in root.handlers)
    has_file = any(isinstance(h, logging.FileHandler) for h in root.handlers)
    if has_console and has_file:
        return

    while root.handlers:
        try:
            h = root.handlers.pop()
            try:
                h.close()
            except Exception:
                pass
        except Exception:
            break

    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    )

    file_path = Path(out_dir) / LOG_FNAME
    fh = logging.FileHandler(filename=file_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    logging.captureWarnings(True)
