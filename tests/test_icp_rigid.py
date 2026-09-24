"""Regression tests for rigid point-cloud registration (fit_rigid)."""
import math

import pytest

from aegisrover.core.types import Pose2, Vec2, wrap_angle
from aegisrover.estimation.icp import centroid, fit_rigid
from aegisrover.geometry.frames import transform_point


def transform_cloud(points, pose):
    return [transform_point(pose, p) for p in points]


# A non-degenerate cloud well away from the origin.
FAR_CLOUD = [Vec2(100.0 + x, -50.0 + y) for x, y in
             [(0.0, 0.0), (1.0, 0.3), (0.5, 1.2), (-0.8, 0.6)]]
NEAR_CLOUD = [Vec2(x, y) for x, y in
              [(0.0, 0.0), (1.0, 0.3), (0.5, 1.2), (-0.8, 0.6)]]


@pytest.mark.parametrize("cloud", [NEAR_CLOUD, FAR_CLOUD])
def test_pure_translation(cloud):
    truth = Pose2(3.5, -2.25, 0.0)
    pose = fit_rigid(cloud, transform_cloud(cloud, truth))
    assert pose.yaw == pytest.approx(0.0, abs=1e-12)
    assert pose.x == pytest.approx(truth.x, abs=1e-10)
    assert pose.y == pytest.approx(truth.y, abs=1e-10)


@pytest.mark.parametrize("cloud", [NEAR_CLOUD, FAR_CLOUD])
def test_pure_rotation_about_origin_gives_zero_translation(cloud):
    truth = Pose2(0.0, 0.0, 0.7)
    pose = fit_rigid(cloud, transform_cloud(cloud, truth))
    assert pose.yaw == pytest.approx(0.7, abs=1e-12)
    assert pose.x == pytest.approx(0.0, abs=1e-10)
    assert pose.y == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("cloud", [NEAR_CLOUD, FAR_CLOUD])
def test_rotation_and_translation(cloud):
    truth = Pose2(10.0, -7.0, 0.6)
    pose = fit_rigid(cloud, transform_cloud(cloud, truth))
    assert wrap_angle(pose.yaw - truth.yaw) == pytest.approx(0.0, abs=1e-12)
    assert pose.x == pytest.approx(truth.x, abs=1e-10)
    assert pose.y == pytest.approx(truth.y, abs=1e-10)


def test_far_cloud_regression_bias_is_origin_independent():
    # The bug biased t by (I - R) @ cd; for a far cloud and ~34.4 deg
    # rotation that error is on the order of tens of metres. It must be
    # recovered identically regardless of cloud distance to origin.
    yaw = 0.6
    truth = Pose2(2.0, 4.0, yaw)
    near = fit_rigid(NEAR_CLOUD, transform_cloud(NEAR_CLOUD, truth))
    far = fit_rigid(FAR_CLOUD, transform_cloud(FAR_CLOUD, truth))
    assert far.x == pytest.approx(near.x, abs=1e-10)
    assert far.y == pytest.approx(near.y, abs=1e-10)
    assert far.yaw == pytest.approx(near.yaw, abs=1e-12)


def test_negative_rotation():
    truth = Pose2(-5.0, 8.0, -1.1)
    pose = fit_rigid(FAR_CLOUD, transform_cloud(FAR_CLOUD, truth))
    assert wrap_angle(pose.yaw - truth.yaw) == pytest.approx(0.0, abs=1e-12)
    assert pose.x == pytest.approx(truth.x, abs=1e-10)
    assert pose.y == pytest.approx(truth.y, abs=1e-10)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        fit_rigid([Vec2(0.0, 0.0)], [Vec2(1.0, 1.0)])
    with pytest.raises(ValueError):
        fit_rigid([Vec2(0.0, 0.0), Vec2(1.0, 0.0)],
                  [Vec2(1.0, 1.0)])


def test_centroid():
    c = centroid([Vec2(0.0, 0.0), Vec2(2.0, 4.0)])
    assert c == Vec2(1.0, 2.0)
