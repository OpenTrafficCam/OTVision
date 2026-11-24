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

_POSITION_WEIGHT = 1 / 20
_VELOCITY_WEIGHT = 1 / 160

_FloatArray = NDArray[np.float64]


class KalmanFilter:
    """Track (center x, center y, width, height) and their per-frame velocities."""

    def __init__(self) -> None:
        self._motion_matrix = np.eye(8)
        self._motion_matrix[:4, 4:] = np.eye(4)

    def initiate(self, measurement: ArrayLike) -> tuple[_FloatArray, _FloatArray]:
        """Return the initial mean (8,) and covariance (8, 8) for a measured box."""
        box = np.asarray(measurement, dtype=np.float64)
        mean = np.concatenate((box, np.zeros_like(box)))
        scale = box[[2, 3, 2, 3]]
        std = np.concatenate(
            (2 * _POSITION_WEIGHT * scale, 10 * _VELOCITY_WEIGHT * scale)
        )
        return mean, np.diag(std**2)

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
        scale = means[:, [2, 3, 2, 3]]
        std = np.concatenate(
            (_POSITION_WEIGHT * scale, _VELOCITY_WEIGHT * scale), axis=1
        )
        motion_covariances = np.eye(8) * std[:, :, None] ** 2
        predicted_means = np.dot(means, self._motion_matrix.T)
        propagated = np.dot(self._motion_matrix, covariances).transpose(1, 0, 2)
        predicted_covariances = (
            np.dot(propagated, self._motion_matrix.T) + motion_covariances
        )
        return predicted_means, predicted_covariances

    def update(
        self, mean: _FloatArray, covariance: _FloatArray, measurement: ArrayLike
    ) -> tuple[_FloatArray, _FloatArray]:
        """Correct a predicted state using a measured (center x, center y, w, h)."""
        measurement_std = _POSITION_WEIGHT * mean[[2, 3, 2, 3]]
        projected_covariance = covariance[:4, :4] + np.diag(measurement_std**2)
        cross_covariance = covariance[:, :4]
        gain = np.linalg.solve(projected_covariance, cross_covariance.T).T

        innovation = np.asarray(measurement, dtype=np.float64) - mean[:4]
        updated_mean = mean + gain.dot(innovation)
        updated_covariance = covariance - gain.dot(projected_covariance).dot(gain.T)
        return updated_mean, updated_covariance
