import os
from datetime import date, datetime
from pathlib import Path

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


def test_expected_slots_rejects_non_divisor():
    import pytest

    with pytest.raises(ValueError, match="divide 1440"):
        expected_slots(7)


def test_complete_full_grid():
    full = _slots(
        date(2026, 6, 3), [(h, m) for h in range(24) for m in (0, 15, 30, 45)]
    )
    assert complete_dates(full, 96) == {date(2026, 6, 3)}


def test_complete_rejects_missing_window():
    full = _slots(
        date(2026, 6, 3), [(h, m) for h in range(24) for m in (0, 15, 30, 45)]
    )
    missing_one = full[:-1]  # drop the 23:45 window with no replacement
    assert complete_dates(missing_one, 96) == set()


def test_complete_tolerates_off_cadence_within_window():
    # a file at 23:47 covers the 23:45 window even though it is off the grid
    full = _slots(
        date(2026, 6, 3), [(h, m) for h in range(24) for m in (0, 15, 30, 45)]
    )
    covered = full[:-1] + _slots(date(2026, 6, 3), [(23, 47)])
    assert complete_dates(covered, 96) == {date(2026, 6, 3)}


def test_complete_tolerates_nonzero_seconds():
    full = _slots(
        date(2026, 6, 3), [(h, m) for h in range(24) for m in (0, 15, 30, 45)]
    )
    full[0] = full[0].replace(second=59)  # drift off :00, still covers window 0
    assert complete_dates(full, 96) == {date(2026, 6, 3)}


def test_complete_tolerates_phase_offset():
    # cam-11 style: every 15 min but phase-shifted to :NN:06 (08, 23, 38, 53)
    base = datetime(2026, 6, 3)
    dts = [
        base.replace(hour=h, minute=m, second=6)
        for h in range(24)
        for m in (8, 23, 38, 53)
    ]
    assert complete_dates(dts, 96) == {date(2026, 6, 3)}


def test_incomplete_when_fewer_than_all_windows():
    full = _slots(
        date(2026, 6, 3), [(h, m) for h in range(24) for m in (0, 15, 30, 45)]
    )
    assert complete_dates(full[:40], 96) == set()


def test_complete_rejects_bad_slots_per_day():
    import pytest

    for bad in (7, 0, -96):
        with pytest.raises(ValueError, match="divide 1440"):
            complete_dates([], bad)


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


def _make_day(cam: Path, day: date, complete=True, old=True, second=0):
    d = cam / f"{day:%Y-%m-%d}"
    d.mkdir(parents=True, exist_ok=True)
    times = [(h, m) for h in range(24) for m in (0, 15, 30, 45)]
    if not complete:
        times = times[:40]
    for h, m in times:
        f = d / f"OTCamera07_FR20_{day:%Y-%m-%d}_{h:02d}-{m:02d}-{second:02d}.otdet"
        f.write_bytes(b"x")
        if old:
            os.utime(f, (1_000_000, 1_000_000))


def test_fires_on_complete_settled_block(tmp_path):
    from otc_coverage import assess_camera

    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    rep = assess_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0),
        tracked_through=None,
        block_days=4,
        slots_per_day=96,
        idle_minutes=5,
    )
    assert rep.fire and rep.tracked_through_after == date(2026, 6, 6)
    assert len(rep.otdet_paths) == 96 * 4


def test_fires_on_second_drifted_block(tmp_path):
    from otc_coverage import assess_camera

    cam = tmp_path / "OTCamera07"
    _make_day(cam, date(2026, 6, 3))
    _make_day(cam, date(2026, 6, 4), second=1)  # whole day off :00 (clock drift)
    _make_day(cam, date(2026, 6, 5), second=2)
    _make_day(cam, date(2026, 6, 6))
    rep = assess_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0),
        tracked_through=None,
        block_days=4,
        slots_per_day=96,
        idle_minutes=5,
    )
    assert rep.fire and rep.tracked_through_after == date(2026, 6, 6)
    assert len(rep.otdet_paths) == 96 * 4


def test_scoped_temp_in_block_blocks_but_stale_outside_does_not(tmp_path):
    from otc_coverage import assess_camera

    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    d7 = cam / "2026-06-07"
    d7.mkdir()
    old = d7 / ".OTCamera07_FR20_2026-06-07_10-00-00.otdet.OLD"
    old.write_bytes(b"")
    os.utime(old, (1_000_000, 1_000_000))
    assert (
        assess_camera(
            cam,
            now=datetime(2026, 6, 9, 12, 0),
            tracked_through=None,
            block_days=4,
            slots_per_day=96,
            idle_minutes=5,
        ).fire
        is True
    )
    new = cam / "2026-06-06" / ".OTCamera07_FR20_2026-06-06_10-30-00.otdet.NEW"
    new.write_bytes(b"")
    rep = assess_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0),
        tracked_through=None,
        block_days=4,
        slots_per_day=96,
        idle_minutes=5,
    )
    assert rep.fire is False and "temp" in rep.reason.lower()


def test_vanished_temp_does_not_raise(tmp_path, monkeypatch):
    from otc_coverage import assess_camera

    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    tmp = cam / "2026-06-06" / ".OTCamera07_FR20_2026-06-06_10-30-00.otdet.NEW"
    tmp.write_bytes(b"")
    real_stat = Path.stat

    def vanish(self, *args, **kwargs):
        if self == tmp:
            raise FileNotFoundError
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", vanish)
    rep = assess_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0),
        tracked_through=None,
        block_days=4,
        slots_per_day=96,
        idle_minutes=5,
    )
    assert rep.fire is True


def test_vanished_window_file_returns_idle(tmp_path, monkeypatch):
    from otc_coverage import assess_camera

    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    gone = cam / "2026-06-06" / "OTCamera07_FR20_2026-06-06_10-30-00.otdet"
    real_stat = Path.stat

    def vanish(self, *args, **kwargs):
        if self == gone:
            raise FileNotFoundError
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", vanish)
    rep = assess_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0),
        tracked_through=None,
        block_days=4,
        slots_per_day=96,
        idle_minutes=5,
    )
    assert rep.fire is False


def test_foreign_host_temp_does_not_block_same_host(tmp_path):
    from otc_coverage import assess_camera

    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    tmp = cam / "2026-06-06" / ".OTCamera09_FR20_2026-06-06_10-30-00.otdet.NEW"
    tmp.write_bytes(b"")
    rep = assess_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0),
        tracked_through=None,
        block_days=4,
        slots_per_day=96,
        idle_minutes=5,
    )
    assert rep.fire is True


def test_foreign_host_ignored(tmp_path):
    from otc_coverage import assess_camera

    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    bad = cam / "2026-06-03" / "OTCamera09_FR20_2026-06-03_00-00-00.otdet"
    bad.write_bytes(b"x")
    os.utime(bad, (1_000_000, 1_000_000))
    rep = assess_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0),
        tracked_through=None,
        block_days=4,
        slots_per_day=96,
        idle_minutes=5,
    )
    assert rep.fire is True
    assert all("OTCamera09" not in p.name for p in rep.otdet_paths)


def test_duplicate_basename_blocks(tmp_path):
    from otc_coverage import assess_camera

    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    dup = "OTCamera07_FR20_2026-06-03_00-00-00.otdet"
    (cam / dup).write_bytes(b"x")
    os.utime(cam / dup, (1_000_000, 1_000_000))
    rep = assess_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0),
        tracked_through=None,
        block_days=4,
        slots_per_day=96,
        idle_minutes=5,
    )
    assert rep.fire is False and "duplicate" in rep.reason.lower()
