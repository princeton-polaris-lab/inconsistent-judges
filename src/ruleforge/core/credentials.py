from __future__ import annotations

import re
import sys
from typing import Sequence

# A URL carrying a credential, read as one token: a user or password in its authority (up to the first /, ?, # or
# space after ://), or a query in the same run of non-space characters, before any fragment.
_CREDENTIAL_URL = re.compile(r"://(?:[^/?#\s]*@|[^#\s]*\?)")
CREDENTIAL_REFUSAL = ("a URL carrying a credential (user:password@ or a query) is never taken, as it would be recorded "
                      "and repeated, so it is not repeated here; give the server's address alone, and name the "
                      "environment variable holding the key with api_key_env")


def carries_credential(text: str) -> bool:
    """Return whether ``text`` holds a URL carrying a credential."""
    return _CREDENTIAL_URL.search(text) is not None


def refused_arguments(arguments: Sequence[str], program: str) -> bool:
    """Return whether a command line holds a URL carrying a credential, saying so on stderr without repeating it."""
    if not any(carries_credential(argument) for argument in arguments):
        return False
    print(f"{program}: {CREDENTIAL_REFUSAL}", file=sys.stderr)
    return True
