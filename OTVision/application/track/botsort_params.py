# Copyright (C) 2022 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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


# Types the pinned Ultralytics release expects per TRACK.BOT_SORT key. A value
# of the wrong type passes YAML parsing but fails deep inside BYTETracker/GMC
# (e.g. TRACK_BUFFER: "90" -> TypeError in max_time_lost; GMC_METHOD: NONE ->
# "Unknown GMC method"), so wrong types are rejected at parse time instead.
_BOTSORT_PARAM_TYPES: dict[str, type] = {
    "track_high_thresh": float,
    "track_low_thresh": float,
    "new_track_thresh": float,
    "match_thresh": float,
    "proximity_thresh": float,
    "appearance_thresh": float,
    "track_buffer": int,
    "fuse_score": bool,
    "with_reid": bool,
    "gmc_method": str,
    "model": str,
}


def normalize_botsort_param_values(
    tracker_params: dict[str, BotSortTrackerParam],
) -> dict[str, BotSortTrackerParam]:
    """Enforce per-key value types and canonicalize enum-like strings.

    ``gmc_method`` is validated case-insensitively, so it is lowered here to
    the form Ultralytics' ``GMC`` accepts. Everything else must already carry
    the expected type; ints are accepted where floats are expected.

    Args:
        tracker_params (dict[str, BotSortTrackerParam]): Normalized-name
            (lowercase-key) BoT-SORT params from YAML.

    Returns:
        dict[str, BotSortTrackerParam]: Params with canonicalized values.

    Raises:
        ValueError: If a value does not have the type Ultralytics expects.
    """
    normalized: dict[str, BotSortTrackerParam] = {}
    for key, value in tracker_params.items():
        expected = _BOTSORT_PARAM_TYPES.get(key)
        if key == "model" and value is None:
            # YAML `MODEL:` parses to None; ReID validation owns its semantics.
            normalized[key] = value
            continue
        if expected is float and isinstance(value, (int, float)):
            if not isinstance(value, bool):
                normalized[key] = float(value)
                continue
        if expected is int and isinstance(value, int) and not isinstance(value, bool):
            normalized[key] = value
            continue
        if expected is bool and isinstance(value, bool):
            normalized[key] = value
            continue
        if expected is str and isinstance(value, str):
            normalized[key] = value.lower() if key == "gmc_method" else value
            continue
        if expected is None:
            normalized[key] = value
            continue
        raise ValueError(
            f"TRACK.BOT_SORT.{key.upper()}={value!r} must be of type "
            f"{expected.__name__}. The pinned Ultralytics release fails on "
            "other types deep inside tracking; fix the YAML value."
        )
    return normalized


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
    """Reject ReID configurations OTVision's detection path cannot satisfy.

    Ultralytics 8.3.159 treats ``model: auto`` as *native detector feature
    tensors* and its encoder calls ``.cpu()`` on them. OTVision supplies NumPy
    frame images, so ``auto`` raises ``AttributeError: 'numpy.ndarray' object
    has no attribute 'cpu'`` inside ``init_track``. An explicit ReID model goes
    through ``ReID(model)``, which does consume images, so it stays allowed.

    A missing, null or empty ``MODEL`` is rejected too: it is not a usable
    model name and must not slip through by failing to equal ``"auto"``.

    Args:
        tracker_params (dict[str, BotSortTrackerParam]): Effective tracker params.

    Raises:
        ValueError: If ReID is enabled without a usable explicit model.
    """
    if not bool(tracker_params.get("with_reid", False)):
        return

    model = tracker_params.get("model")
    model_name = "" if model is None else str(model).strip().lower()
    if model_name in {"", "auto", "none", "null"}:
        raise ValueError(
            "BoT-SORT ReID needs an explicit MODEL. Ultralytics 8.3.159 reads "
            f"MODEL={model!r} as native detector feature tensors, but OTVision "
            "supplies NumPy frame images, so tracking would fail inside "
            "Ultralytics. Set an explicit ReID model path/name, or disable ReID "
            "(`WITH_REID: false`)."
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
