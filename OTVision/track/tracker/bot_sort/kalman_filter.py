# Adapted from NirAharon/BoT-SORT, tracker/kalman_filter.py
# https://github.com/NirAharon/BoT-SORT/blob/f98fdbf21af6c45ccd9010b842c0cc86b964df84/tracker/kalman_filter.py
#
# MIT License
#
# Copyright (c) 2022 Nir Aharon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from numpy.typing import ArrayLike, NDArray

BOX_DIMENSIONS = 4
"""Box coordinates (center x, center y, width, height) in the state."""
STATE_DIMENSIONS = 2 * BOX_DIMENSIONS
"""The box followed by the per-frame velocity of each box coordinate."""
STATE_BOX = slice(0, BOX_DIMENSIONS)
STATE_VELOCITY = slice(BOX_DIMENSIONS, STATE_DIMENSIONS)

_SCALE_INDICES = [2, 3, 2, 3]
"""Box size (width, height, width, height) that scales the uncertainty of each
box coordinate: center x and width scale with the width, center y and height with
the height. Large boxes move more pixels per frame and have larger pixel errors."""

_POSITION_WEIGHT = 1 / 20
"""Standard deviation of a box coordinate, as a fraction of the box size."""
_VELOCITY_WEIGHT = 1 / 160
"""Standard deviation of a box velocity per frame, as a fraction of the box size."""
# A new track starts less certain than the noise added per frame, the velocity
# much more so, because a single box gives no velocity estimate.
_INITIAL_POSITION_FACTOR = 2
_INITIAL_VELOCITY_FACTOR = 10

_FloatArray = NDArray[np.float64]


def _diagonal_covariances(std: _FloatArray) -> _FloatArray:
    """Return covariances (..., n, n) of independent errors with std (..., n)."""
    return std[..., :, None] ** 2 * np.eye(std.shape[-1])


class KalmanFilter:
    """Track (center x, center y, width, height) and their per-frame velocities."""

    def __init__(self) -> None:
        # Constant velocity: each box coordinate advances by its velocity per frame.
        self._motion_matrix = np.eye(STATE_DIMENSIONS)
        self._motion_matrix[STATE_BOX, STATE_VELOCITY] = np.eye(BOX_DIMENSIONS)

    def initiate(self, measurement: ArrayLike) -> tuple[_FloatArray, _FloatArray]:
        """Return the initial mean (8,) and covariance (8, 8) for a measured box."""
        box = np.asarray(measurement, dtype=np.float64)
        mean = np.concatenate((box, np.zeros_like(box)))
        scale = box[_SCALE_INDICES]
        std = np.concatenate(
            (
                _INITIAL_POSITION_FACTOR * _POSITION_WEIGHT * scale,
                _INITIAL_VELOCITY_FACTOR * _VELOCITY_WEIGHT * scale,
            )
        )
        return mean, _diagonal_covariances(std)

    def predict(
        self, mean: _FloatArray, covariance: _FloatArray
    ) -> tuple[_FloatArray, _FloatArray]:
        """Return the mean and covariance one frame ahead."""
        means, covariances = self.multi_predict(mean[None, :], covariance[None, :, :])
        return means[0], covariances[0]

    def multi_predict(
        self, means: _FloatArray, covariances: _FloatArray
    ) -> tuple[_FloatArray, _FloatArray]:
        """Predict one frame ahead for means (N, 8) and covariances (N, 8, 8)."""
        scale = means[:, _SCALE_INDICES]
        std = np.concatenate(
            (_POSITION_WEIGHT * scale, _VELOCITY_WEIGHT * scale), axis=1
        )
        motion = self._motion_matrix
        # np.dot instead of @: on Apple Silicon, numpy's matmul reports spurious
        # divide-by-zero, overflow and invalid-value warnings for these products.
        # The values are the same either way.
        predicted_means = np.dot(means, motion.T)
        propagated = np.dot(motion, covariances).transpose(1, 0, 2)
        predicted_covariances = np.dot(propagated, motion.T) + _diagonal_covariances(
            std
        )
        return predicted_means, predicted_covariances

    def update(
        self, mean: _FloatArray, covariance: _FloatArray, measurement: ArrayLike
    ) -> tuple[_FloatArray, _FloatArray]:
        """Correct a predicted state using a measured (center x, center y, w, h)."""
        measurement_std = _POSITION_WEIGHT * mean[_SCALE_INDICES]
        projected_covariance = covariance[STATE_BOX, STATE_BOX] + _diagonal_covariances(
            measurement_std
        )
        cross_covariance = covariance[:, STATE_BOX]
        gain = np.linalg.solve(projected_covariance, cross_covariance.T).T

        innovation = np.asarray(measurement, dtype=np.float64) - mean[STATE_BOX]
        updated_mean = mean + gain @ innovation
        updated_covariance = covariance - gain @ projected_covariance @ gain.T
        return updated_mean, updated_covariance
