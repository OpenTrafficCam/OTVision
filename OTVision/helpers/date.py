from datetime import datetime, timezone


def parse_date_string_to_utc_datime(date_string: str, date_format: str) -> datetime:
    """Parse a date string to a datetime object with UTC set as timezone.

    Args:
        date_string (str): the date string
        date_format (str): the date format

    Returns:
        datetime: the datetime object with UTC as set timezone
    """
    return datetime.strptime(date_string, date_format).replace(tzinfo=timezone.utc)


def parse_timestamp_string_to_utc_datetime(timestamp: str | float) -> datetime:
    """Parse timestamp string to  datetime object with UTC set as timezone.

    Args:
        timestamp (str | float): the timestamp string to be parsed

    Returns:
        datetime: the datetime object with UTC as set timezone
    """
    return datetime.fromtimestamp(float(timestamp), timezone.utc)
