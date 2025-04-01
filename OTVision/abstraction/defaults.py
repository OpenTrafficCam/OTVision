def value_or_default[T](value: T | None, default: T) -> T:
    """
    Returns the provided value if it is not None; otherwise, returns a default value.

    Args:
        value (T | None): The value to be evaluated.
        default (T): The fallback value if 'value' is None.

    Returns:
        T: The provided value or the default value if the original is None.
    """

    if value is not None:
        return value
    return default
