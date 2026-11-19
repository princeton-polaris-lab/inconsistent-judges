from __future__ import annotations

import sys
from typing import Optional, Sequence

from ruleforge.experiments.training.pins import PINNED_PROTECTED
from ruleforge.grpo import grpo_data
from ruleforge.launch.programs import check_top


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the GRPO rows build under the paper's pins, from the checkout's top only."""
    check_top()
    return grpo_data.main(argv, pins=PINNED_PROTECTED)


if __name__ == "__main__":
    sys.exit(main())
