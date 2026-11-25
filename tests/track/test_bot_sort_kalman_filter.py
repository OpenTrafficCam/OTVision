"""Tests for the BoT-SORT Kalman filter.

Expected values are worked out by hand from the filter parameters. The standard
deviation of each box coordinate is a fraction of the box size: width for center x
and width, height for center y and height.

- initiate: 2/20 of the size for the box, 10/160 for the velocity
- predict, motion noise: 1/20 of the size for the box, 1/160 for the velocity
- update, measurement noise: 1/20 of the size
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from OTVision.track.tracker.bot_sort.kalman_filter import (
    STATE_BOX,
    STATE_VELOCITY,
    KalmanFilter,
)

BOX = np.array([100.0, 50.0, 40.0, 20.0])
"""Center (100, 50), 40 wide and 20 high."""
BOX_VELOCITY = np.array([3.0, -2.0, 1.0, -1.0])


def _covariance(
    box_variance: NDArray[np.float64],
    velocity_variance: NDArray[np.float64],
    box_velocity_covariance: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Build an 8x8 covariance where each box coordinate correlates only with its
    own velocity."""
    covariance = np.diag(np.concatenate((box_variance, velocity_variance)))
    if box_velocity_covariance is not None:
        covariance[STATE_BOX, STATE_VELOCITY] = np.diag(box_velocity_covariance)
        covariance[STATE_VELOCITY, STATE_BOX] = np.diag(box_velocity_covariance)
    return covariance


# Variances for BOX (width 40, height 20), from the std fractions in the module
# docstring.
INITIAL_BOX_VARIANCE = np.square([4.0, 2.0, 4.0, 2.0])
INITIAL_VELOCITY_VARIANCE = np.square([2.5, 1.25, 2.5, 1.25])
MOTION_BOX_VARIANCE = np.square([2.0, 1.0, 2.0, 1.0])
MOTION_VELOCITY_VARIANCE = np.square([0.25, 0.125, 0.25, 0.125])

# One frame after initiate: the box variance takes up the velocity variance
# (box + velocity), and both get the motion noise.
PREDICTED_BOX_VARIANCE = (
    INITIAL_BOX_VARIANCE + INITIAL_VELOCITY_VARIANCE + MOTION_BOX_VARIANCE
)
PREDICTED_VELOCITY_VARIANCE = INITIAL_VELOCITY_VARIANCE + MOTION_VELOCITY_VARIANCE
PREDICTED_BOX_VELOCITY_COVARIANCE = INITIAL_VELOCITY_VARIANCE


def test_initiate_starts_at_measured_box_with_zero_velocity() -> None:
    kalman_filter = KalmanFilter()

    mean, covariance = kalman_filter.initiate(BOX)

    np.testing.assert_array_equal(mean[STATE_BOX], BOX)
    np.testing.assert_array_equal(mean[STATE_VELOCITY], np.zeros(4))
    np.testing.assert_array_equal(
        covariance, _covariance(INITIAL_BOX_VARIANCE, INITIAL_VELOCITY_VARIANCE)
    )


def test_predict_advances_one_frame_and_adds_motion_uncertainty() -> None:
    kalman_filter = KalmanFilter()
    _, covariance = kalman_filter.initiate(BOX)
    mean = np.concatenate((BOX, BOX_VELOCITY))

    predicted_mean, predicted_covariance = kalman_filter.predict(mean, covariance)

    np.testing.assert_array_equal(predicted_mean[STATE_BOX], BOX + BOX_VELOCITY)
    np.testing.assert_array_equal(predicted_mean[STATE_VELOCITY], BOX_VELOCITY)
    np.testing.assert_array_equal(
        predicted_covariance,
        _covariance(
            PREDICTED_BOX_VARIANCE,
            PREDICTED_VELOCITY_VARIANCE,
            PREDICTED_BOX_VELOCITY_COVARIANCE,
        ),
    )


def test_multi_predict_advances_each_track_with_its_own_box_scale() -> None:
    kalman_filter = KalmanFilter()
    # 10 wide and 60 high, so its uncertainty scales differently from BOX.
    second_box = np.array([40.0, 80.0, 10.0, 60.0])
    second_velocity = np.array([-4.0, 6.0, -2.0, 2.0])
    first_mean, first_covariance = kalman_filter.initiate(BOX)
    second_mean, second_covariance = kalman_filter.initiate(second_box)
    first_mean[STATE_VELOCITY] = BOX_VELOCITY
    second_mean[STATE_VELOCITY] = second_velocity
    means = np.stack((first_mean, second_mean))
    covariances = np.stack((first_covariance, second_covariance))

    predicted_means, predicted_covariances = kalman_filter.multi_predict(
        means, covariances
    )

    np.testing.assert_array_equal(
        predicted_means,
        [
            np.concatenate((BOX + BOX_VELOCITY, BOX_VELOCITY)),
            np.concatenate((second_box + second_velocity, second_velocity)),
        ],
    )
    # Same steps as PREDICTED_*_VARIANCE, with the std fractions applied to
    # width 10 and height 60.
    second_initial_box_variance = np.square([1.0, 6.0, 1.0, 6.0])
    second_initial_velocity_variance = np.square([0.625, 3.75, 0.625, 3.75])
    second_motion_box_variance = np.square([0.5, 3.0, 0.5, 3.0])
    second_motion_velocity_variance = np.square([0.0625, 0.375, 0.0625, 0.375])
    np.testing.assert_array_equal(
        np.diagonal(predicted_covariances, axis1=1, axis2=2),
        [
            np.concatenate((PREDICTED_BOX_VARIANCE, PREDICTED_VELOCITY_VARIANCE)),
            np.concatenate(
                (
                    second_initial_box_variance
                    + second_initial_velocity_variance
                    + second_motion_box_variance,
                    second_initial_velocity_variance + second_motion_velocity_variance,
                )
            ),
        ],
    )


def test_update_corrects_position_velocity_and_uncertainty() -> None:
    kalman_filter = KalmanFilter()
    mean, covariance = kalman_filter.initiate(BOX)
    predicted_mean, predicted_covariance = kalman_filter.predict(mean, covariance)
    measured_box = np.array([106.0, 47.0, 48.0, 16.0])

    updated_mean, updated_covariance = kalman_filter.update(
        predicted_mean, predicted_covariance, measured_box
    )

    # Measurement variance is (1/20 of the size)^2 = [4, 1, 4, 1]. For center x,
    # the predicted box variance is 26.25, so the residual variance is
    # 26.25 + 4 = 30.25 = 121/4. Every coordinate scales the same way with the
    # box size, so all four share the same gains:
    #   box gain      = 26.25 / 30.25 = 105/121
    #   velocity gain =  6.25 / 30.25 =  25/121
    box_gain = 105 / 121
    velocity_gain = 25 / 121
    innovation = measured_box - BOX
    np.testing.assert_allclose(
        updated_mean[STATE_BOX], BOX + box_gain * innovation, rtol=1e-12
    )
    np.testing.assert_allclose(
        updated_mean[STATE_VELOCITY], velocity_gain * innovation, rtol=1e-12
    )
    # The box variance and the box-velocity covariance shrink by (1 - box gain).
    # The velocity variance drops by the velocity gain times the box-velocity
    # covariance.
    np.testing.assert_allclose(
        updated_covariance,
        _covariance(
            box_variance=PREDICTED_BOX_VARIANCE * (1 - box_gain),
            velocity_variance=PREDICTED_VELOCITY_VARIANCE
            - velocity_gain * PREDICTED_BOX_VELOCITY_COVARIANCE,
            box_velocity_covariance=PREDICTED_BOX_VELOCITY_COVARIANCE * (1 - box_gain),
        ),
        rtol=1e-12,
    )


MEASURED_FRAMES = 5


@pytest.mark.parametrize(
    ("velocity", "expected_box"),
    [
        pytest.param((0.0, 0.0), BOX, id="stationary"),
        # The true center one frame after the last measurement is
        # (100, 50) + 6 * (4, -2) = (124, 38). After five measurements the
        # velocity estimate still lags, so the prediction falls short of it.
        # Regression value, not derived by hand. The size does not move.
        pytest.param(
            (4.0, -2.0),
            [122.99125201923215, 38.504373990383925, 40.0, 20.0],
            id="constant-speed",
        ),
    ],
)
def test_predict_follows_box_motion_over_multiple_frames(
    velocity: tuple[float, float], expected_box: NDArray[np.float64]
) -> None:
    kalman_filter = KalmanFilter()
    box_velocity = np.array([*velocity, 0.0, 0.0])
    mean, covariance = kalman_filter.initiate(BOX)
    for frame in range(1, MEASURED_FRAMES + 1):
        mean, covariance = kalman_filter.predict(mean, covariance)
        mean, covariance = kalman_filter.update(
            mean, covariance, BOX + frame * box_velocity
        )

    predicted_mean, _ = kalman_filter.predict(mean, covariance)

    np.testing.assert_allclose(
        predicted_mean[STATE_BOX], expected_box, rtol=0, atol=1e-12
    )
