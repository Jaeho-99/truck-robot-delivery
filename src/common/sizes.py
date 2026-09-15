"""Single source of truth for supported customer counts and result folders."""

import re

SUPPORTED_SIZES = (5, 10, 20, 50, 100, 150, 200, 250, 300)
SIZE_PATTERN = re.compile(r"n(" + "|".join(map(str, SUPPORTED_SIZES)) + r")\Z")
