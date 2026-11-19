def diff(before, after, prefix=""):
    """The keys whose values differ between two documents, dotted, with each side's value."""
    if isinstance(before, dict) and isinstance(after, dict):
        return {key: value for name in sorted(set(before) | set(after))
                for key, value in diff(before.get(name), after.get(name), f"{prefix}{name}.").items()}
    return {} if before == after else {prefix[:-1]: (before, after)}
