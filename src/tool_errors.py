"""Errors that tell the calling agent what was wrong with its input and how to fix it."""


class ToolInputError(Exception):
    """A mistake in the caller's arguments. The payload tells the agent exactly what was wrong,
    how to fix it, and the rules/valid values it needs to get it right on the next call."""

    def __init__(self, error: str, fix: str, **context):
        super().__init__(error)
        self.payload = {"error": error, "fix": fix, **context}
