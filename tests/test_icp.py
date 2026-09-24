import math

import pytest

from aegisrover.core.types import Pose2, Vec2
from aegisrover.estimation.icp import fit_rigid


def apply_pose(pose: Pose2, point: Vec2) -> Vec2:
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    return Vec2(
        pose.x + c * point.x - s * point.y,
        pose.y + s * point.x + c * point.y,
    )


@pytest.mark.parametrize(
    'expected',
    [
        Pose2(5.0, -3.0, 0.0),
        Pose2(-37.0, 19.0, 0.37),
        Pose2(120.0, -80.0, -2.1),
    ],
)
def test_rigid_fit_recovers_rotation_and_translation_far_from_origin(expected):
    src = [
        Vec2(100.0, 10.0),
        Vec2(102.0, 12.0),
        Vec2(98.0, -4.0),
        Vec2(105.0, 3.0),
    ]
    dst = [apply_pose(expected, point) for point in src]

    estimated = fit_rigid(src, dst)

    assert estimated.yaw == pytest.approx(expected.yaw, abs=1e-12)
    assert estimated.x == pytest.approx(expected.x, abs=1e-12)
    assert estimated.y == pytest.approx(expected.y, abs=1e-12)


def test_rigid_fit_still_handles_nearby_small_cloud():
    src = [Vec2(0.10, 0.02), Vec2(0.12, 0.04), Vec2(0.08, -0.01)]
    expected = Pose2(-0.03, 0.07, 0.25)
    dst = [apply_pose(expected, point) for point in src]

    estimated = fit_rigid(src, dst)

    assert estimated.yaw == pytest.approx(expected.yaw, abs=1e-12)
    assert estimated.x == pytest.approx(expected.x, abs=1e-12)
    assert estimated.y == pytest.approx(expected.y, abs=1e-12)
