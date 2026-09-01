"""Resolution and validation of BoT-SORT tracker parameters.

Pure configuration logic: turns ``TRACK.BOT_SORT`` plus a source frame rate
into the effective Ultralytics arguments, and rejects settings the pinned
Ultralytics release cannot honour. Kept separate from the tracker plugin so
metadata writers and the config parser need not depend on the tracker.
"""

import logging
from math import ceil

from OTVision import dataformat
from OTVision.application.config import (
    DEFAULT_BOTSORT_TRACKER_PARAMS,
    BotSortTrackerParam,
    TrackConfig,
    _TrackBotSortConfig,
)
from OTVision.domain.tracker import TrackerType
from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


def _try_positive_float(value: object) -> float | None:
    """Return *value* as a positive float, or ``None`` if conversion fails.

    Args:
        value (object): Candidate numeric value (int, float, Decimal, or
            numeric string).

    Returns:
        float | None: Positive float, or ``None`` when conversion fails.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        # ``.otdet`` metadata may store FPS as Decimal via ijson.
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def extract_frame_rate_from_metadata(metadata: dict) -> float | None:
    """Extract preferred frame rate from OTDET metadata.

    Prefers ``actual_fps`` over ``recorded_fps`` when both are present and positive.

    Args:
        metadata (dict): OTDET metadata dictionary.

    Returns:
        float | None: Positive FPS value, or ``None`` if unavailable.
    """
    video = metadata.get(dataformat.VIDEO, {})
    if not isinstance(video, dict):
        return None

    actual = _try_positive_float(video.get(dataformat.ACTUAL_FPS))
    if actual is not None:
        return actual

    recorded = _try_positive_float(video.get(dataformat.RECORDED_FPS))
    if recorded is not None:
        return recorded

    return None


def ultralytics_effective_miss_frames(frame_rate: int, track_buffer: int) -> int:
    """Compute Ultralytics' occlusion window for a ``track_buffer``.

    Ultralytics uses ``int(fps / 30 * track_buffer)``.

    Args:
        frame_rate (int): Video frame rate used by BOTSORT.
        track_buffer (int): Ultralytics ``track_buffer`` argument.

    Returns:
        int: Effective number of missing frames before Ultralytics drops a track.
    """
    fps = max(1, int(frame_rate))
    return int(fps / 30 * int(track_buffer))


def derive_track_buffer(t_miss_max: int, frame_rate: int) -> int:
    """Derive Ultralytics ``track_buffer`` covering at least ``t_miss_max``.

    Ultralytics floors ``int(fps / 30 * track_buffer)``, so ``round`` is not an inverse.
    Using ``ceil(t_miss_max * 30 / fps)`` guarantees the effective lifetime is at least
    ``t_miss_max``. Exact equality is impossible for some FPS values.

    Args:
        t_miss_max (int): OTVision missing-frame threshold.
        frame_rate (int): Video frame rate used by BOTSORT.

    Returns:
        int: Derived ``track_buffer`` (>= 1).
    """
    fps = max(1, int(frame_rate))
    buffer = max(1, int(ceil(t_miss_max * 30 / fps)))
    # Guard floating-point edge cases so effective lifetime never undershoots.
    while ultralytics_effective_miss_frames(fps, buffer) < t_miss_max:
        buffer += 1
    return buffer


def resolve_botsort_tracker_params(
    botsort_config: _TrackBotSortConfig,
    frame_rate: int,
) -> dict[str, BotSortTrackerParam]:
    """Resolve effective BoT-SORT args (defaults + overrides + derived buffer).

    This is the single source of truth for tracker construction and ``.ottrk``
    metadata. When YAML does not set ``track_buffer``, it is derived from
    ``t_miss_max`` and ``frame_rate``.

    Args:
        botsort_config (_TrackBotSortConfig): BoT-SORT configuration section.
        frame_rate (int): Video frame rate used by BOTSORT.

    Returns:
        dict[str, BotSortTrackerParam]: Effective Ultralytics tracker arguments.
    """
    args_dict: dict[str, BotSortTrackerParam] = dict(DEFAULT_BOTSORT_TRACKER_PARAMS)
    args_dict.update(botsort_config.tracker_params)

    derived_buffer = derive_track_buffer(botsort_config.t_miss_max, frame_rate)
    if "track_buffer" not in botsort_config.tracker_params:
        args_dict["track_buffer"] = derived_buffer
    else:
        explicit_buffer = int(args_dict["track_buffer"])
        effective_miss = ultralytics_effective_miss_frames(frame_rate, explicit_buffer)
        # Only an UNDERSIZED window is harmful: Ultralytics would drop the track
        # before OTVision's own T_MISS_MAX, splitting one object into several OT
        # ids. A larger window is harmless because OTVision still evicts first.
        if effective_miss < botsort_config.t_miss_max:
            log.warning(
                "TRACK.BOT_SORT.TRACK_BUFFER=%s at %s fps implies an Ultralytics "
                "occlusion window of only %s frames, shorter than T_MISS_MAX=%s. "
                "Ultralytics will drop tracks before OTVision does, splitting one "
                "physical object into multiple OTVision track IDs. "
                "Omit TRACK_BUFFER to derive %s automatically.",
                explicit_buffer,
                frame_rate,
                effective_miss,
                botsort_config.t_miss_max,
                derived_buffer,
            )

    return args_dict


def resolve_botsort_params_for_fps(
    track_config: TrackConfig,
    fps: float | None,
) -> dict[str, BotSortTrackerParam]:
    """Resolve effective BoT-SORT params for metadata, given a raw FPS.

    Returns an empty mapping when BoT-SORT is not the selected tracker, so
    metadata writers need no tracker-type branch of their own.

    Args:
        track_config (TrackConfig): Track configuration.
        fps (float | None): Frame rate of the source, if known.

    Returns:
        dict[str, BotSortTrackerParam]: Effective params, or empty for other trackers.

    Raises:
        ValueError: If BoT-SORT is selected but ``fps`` is missing or not positive.
    """
    if track_config.tracker_type is not TrackerType.BOTSORT:
        return {}
    if fps is None or fps <= 0:
        raise ValueError(
            "BoT-SORT requires a positive source FPS to resolve effective "
            f"tracker parameters, but got {fps!r}."
        )
    return resolve_botsort_tracker_params(track_config.botsort, to_frame_rate(fps))


def to_frame_rate(fps: float) -> int:
    """Round a raw FPS to the positive integer frame rate BOTSORT is built with.

    Args:
        fps (float): Raw frame rate from source metadata.

    Returns:
        int: Rounded frame rate, at least 1.
    """
    return max(1, int(round(fps)))


def validate_botsort_reid_config(
    tracker_params: dict[str, BotSortTrackerParam],
) -> None:
    """Reject ReID, which the pinned Ultralytics release cannot perform.

    Ultralytics 8.3.159 sets ``BOTSORT.encoder = None`` whenever ``with_reid``
    is requested ("Haven't supported BoT-SORT(reid) yet"), and only uses ReID
    features when that encoder is non-null. Enabling ReID therefore changes
    nothing regardless of ``model``, so accepting it would silently promise
    appearance matching that never happens.

    Args:
        tracker_params (dict[str, BotSortTrackerParam]): Effective tracker params.

    Raises:
        ValueError: If ReID is enabled.
    """
    if bool(tracker_params.get("with_reid", False)):
        raise ValueError(
            "BoT-SORT ReID is not supported: Ultralytics 8.3.159 disables its "
            "ReID encoder unconditionally, so WITH_REID has no effect. "
            "Set `WITH_REID: false` under TRACK.BOT_SORT."
        )


def validate_botsort_gmc_config(
    tracker_params: dict[str, BotSortTrackerParam],
) -> None:
    """Reject global motion compensation, which needs frame images.

    Ultralytics silently skips GMC when no image is supplied, and OTVision's
    ``.otdet`` input carries detections only. A non-``none`` ``GMC_METHOD``
    would therefore advertise camera-motion correction that never runs.

    Args:
        tracker_params (dict[str, BotSortTrackerParam]): Effective tracker params.

    Raises:
        ValueError: If a GMC method other than ``none`` is configured.
    """
    gmc_method = str(tracker_params.get("gmc_method", "none")).lower()
    if gmc_method != "none":
        raise ValueError(
            f"BoT-SORT GMC_METHOD='{gmc_method}' requires frame images, but "
            "OTVision tracks from .otdet detections without images, so "
            "Ultralytics would skip motion compensation silently. "
            "Set `GMC_METHOD: none` under TRACK.BOT_SORT."
        )
