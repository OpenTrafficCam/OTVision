from datetime import datetime, timezone

from OTVision.dataformat import DATE_FORMAT


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


def parse_datetime(date: str | float) -> datetime:
    """Parse a date string or timestamp to a datetime with UTC as timezone.

    Args:
        date (str | float): the date to parse

    Returns:
        datetime: the parsed datetime object with UTC set as timezone
    """
    if isinstance(date, str) and ("-" in date):
        return parse_date_string_to_utc_datime(date, DATE_FORMAT)
    return parse_timestamp_string_to_utc_datetime(date)
