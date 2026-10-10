"""Errors raised by the AI layer. Messages are safe to show to the user."""


class LLMError(Exception):
    """The language model could not produce an answer."""