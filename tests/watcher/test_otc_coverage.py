from datetime import date, datetime

from otc_coverage import (
    complete_dates,
    consecutive_runs,
    expected_slots,
    next_block,
    parse_otc_filename,
)


def test_parse_valid():
    assert parse_otc_filename("OTCamera07_FR20_2026-06-03_00-15-00.otdet") == (
        "OTCamera07",
        datetime(2026, 6, 3, 0, 15, 0),
    )


def test_parse_rejects_junk():
    assert parse_otc_filename("._OTCamera07_FR20_2026-06-03_00-15-00.otdet") is None
    assert (
        parse_otc_filename("OTCamera07_FR20_2026-06-03_00-15-00.otdet.TMP9")
        is None
    )
    assert parse_otc_filename("notes.txt") is None


def _slots(day, times):
    base = datetime(day.year, day.month, day.day)
    return [base.replace(hour=h, minute=m) for h, m in times]


def test_expected_slots_15min():
    s = expected_slots(96)
    assert (0, 0) in s and (23, 45) in s and len(s) == 96 and (0, 7) not in s


def test_complete_requires_exact_set():
    full = _slots(
        date(2026, 6, 3), [(h, m) for h in range(24) for m in (0, 15, 30, 45)]
    )
    missing_one = full[:-1]
    off_cadence = full[:-1] + _slots(date(2026, 6, 3), [(23, 47)])
    assert complete_dates(full, 96) == {date(2026, 6, 3)}
    assert complete_dates(missing_one, 96) == set()
    assert complete_dates(off_cadence, 96) == set()


def test_runs():
    ds = {date(2026, 6, 3), date(2026, 6, 4), date(2026, 6, 6)}
    assert consecutive_runs(ds) == [
        [date(2026, 6, 3), date(2026, 6, 4)],
        [date(2026, 6, 6)],
    ]


D = lambda n: date(2026, 6, n)  # noqa: E731


def test_block_needs_full_block_days():
    assert next_block({D(3), D(4), D(5)}, None, 4) == []
    assert next_block({D(3), D(4), D(5), D(6)}, None, 4) == [
        D(3),
        D(4),
        D(5),
        D(6),
    ]


def test_non_overlapping_next_block():
    run8 = {D(n) for n in range(3, 11)}
    assert next_block(run8, tracked_through=D(6), block_days=4) == [
        D(7),
        D(8),
        D(9),
        D(10),
    ]


def test_leftover_smaller_than_block_waits():
    run6 = {D(n) for n in range(3, 9)}
    assert next_block(run6, tracked_through=D(6), block_days=4) == []
