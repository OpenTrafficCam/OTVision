import numpy as np
import pytest

from OTVision.track.tracker.bot_sort.kalman_filter import KalmanFilter


def test_initiate_starts_at_measured_box_with_zero_velocity() -> None:
    kalman_filter = KalmanFilter()

    mean, covariance = kalman_filter.initiate(np.array([100.0, 50.0, 40.0, 20.0]))

    np.testing.assert_array_equal(mean, [100.0, 50.0, 40.0, 20.0, 0, 0, 0, 0])
    np.testing.assert_array_equal(
        covariance, np.diag([16.0, 4.0, 16.0, 4.0, 6.25, 1.5625, 6.25, 1.5625])
    )


def test_predict_advances_one_frame_and_adds_motion_uncertainty() -> None:
    kalman_filter = KalmanFilter()
    _, covariance = kalman_filter.initiate([100.0, 50.0, 40.0, 20.0])
    mean = np.array([100.0, 50.0, 40.0, 20.0, 3.0, -2.0, 1.0, -1.0])

    predicted_mean, predicted_covariance = kalman_filter.predict(mean, covariance)

    np.testing.assert_array_equal(
        predicted_mean, [103.0, 48.0, 41.0, 19.0, 3.0, -2.0, 1.0, -1.0]
    )
    cross_covariance = [6.25, 1.5625, 6.25, 1.5625]
    expected_covariance = (
        np.diag([26.25, 6.5625, 26.25, 6.5625, 6.3125, 1.578125, 6.3125, 1.578125])
        + np.diag(cross_covariance, 4)
        + np.diag(cross_covariance, -4)
    )
    np.testing.assert_array_equal(predicted_covariance, expected_covariance)


def test_multi_predict_advances_each_track_with_its_own_box_scale() -> None:
    kalman_filter = KalmanFilter()
    first_mean, first_covariance = kalman_filter.initiate([100.0, 50.0, 40.0, 20.0])
    second_mean, second_covariance = kalman_filter.initiate([40.0, 80.0, 10.0, 60.0])
    first_mean[4:] = [3.0, -2.0, 1.0, -1.0]
    second_mean[4:] = [-4.0, 6.0, -2.0, 2.0]
    means = np.stack((first_mean, second_mean))
    covariances = np.stack((first_covariance, second_covariance))
    predicted_means, predicted_covariances = kalman_filter.multi_predict(
        means, covariances
    )

    np.testing.assert_array_equal(
        predicted_means,
        [[103, 48, 41, 19, 3, -2, 1, -1], [36, 86, 8, 62, -4, 6, -2, 2]],
    )
    np.testing.assert_array_equal(
        np.diagonal(predicted_covariances, axis1=1, axis2=2),
        [
            [26.25, 6.5625, 26.25, 6.5625, 6.3125, 1.578125, 6.3125, 1.578125],
            [
                1.640625,
                59.0625,
                1.640625,
                59.0625,
                0.39453125,
                14.203125,
                0.39453125,
                14.203125,
            ],
        ],
    )


def test_update_corrects_position_velocity_and_uncertainty() -> None:
    kalman_filter = KalmanFilter()
    mean, covariance = kalman_filter.initiate([100.0, 50.0, 40.0, 20.0])
    predicted_mean, predicted_covariance = kalman_filter.predict(mean, covariance)

    updated_mean, updated_covariance = kalman_filter.update(
        predicted_mean, predicted_covariance, [106.0, 47.0, 48.0, 16.0]
    )

    np.testing.assert_allclose(
        updated_mean,
        [
            105.20661157024793,
            47.396694214876035,
            46.94214876033058,
            16.52892561983471,
            1.2396694214876034,
            -0.6198347107438017,
            1.6528925619834711,
            -0.8264462809917356,
        ],
        rtol=1e-12,
    )
    position_variance = [3.4710743801652892, 0.8677685950413223] * 2
    velocity_variance = [5.021177685950414, 1.2552944214876034] * 2
    cross_covariance = [0.8264462809917356, 0.2066115702479339] * 2
    expected_covariance = (
        np.diag(position_variance + velocity_variance)
        + np.diag(cross_covariance, 4)
        + np.diag(cross_covariance, -4)
    )
    np.testing.assert_allclose(updated_covariance, expected_covariance, rtol=1e-12)


@pytest.mark.parametrize(
    ("velocity", "expected_box"),
    [
        pytest.param((0.0, 0.0), [100.0, 50.0, 40.0, 20.0], id="stationary"),
        pytest.param(
            (4.0, -2.0),
            [122.99125201923215, 38.504373990383925, 40.0, 20.0],
            id="constant-speed",
        ),
    ],
)
def test_predict_follows_box_motion_over_multiple_frames(
    velocity: tuple[float, float], expected_box: list[float]
) -> None:
    kalman_filter = KalmanFilter()
    first_box = np.array([100.0, 50.0, 40.0, 20.0])
    box_velocity = np.array([*velocity, 0.0, 0.0])
    mean, covariance = kalman_filter.initiate(first_box)
    for frame in range(1, 6):
        mean, covariance = kalman_filter.predict(mean, covariance)
        mean, covariance = kalman_filter.update(
            mean, covariance, first_box + frame * box_velocity
        )

    predicted_mean, _ = kalman_filter.predict(mean, covariance)

    np.testing.assert_allclose(predicted_mean[:4], expected_box, rtol=0, atol=1e-12)
