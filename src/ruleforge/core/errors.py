class RuleForgeError(Exception):
    """The base of the library's errors: ``except RuleForgeError`` catches any of them."""


class TracebackError(RuleForgeError):
    """An error a command ends on with its traceback, where the command line refuses other errors in one line."""


class SamplingDesignError(ValueError, RuleForgeError):
    """Raised when a campaign identity, config or manifest is unusable."""


class ArtifactReadError(ValueError, RuleForgeError):
    """Raised when a frozen artifact cannot be interpreted."""


class ArtifactRootError(RuntimeError, RuleForgeError):
    """Raised when the artifact root is unset or is not a directory."""
