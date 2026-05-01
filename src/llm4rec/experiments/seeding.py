"""Central random seed setup."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> int:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    return seed
