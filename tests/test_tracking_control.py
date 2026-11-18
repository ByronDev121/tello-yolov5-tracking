import unittest

from tracking_control import (
    FACE_PROFILE,
    Detection,
    compute_command,
)


class ComputeCommandTests(unittest.TestCase):
    def test_first_detection_has_no_derivative_kick(self):
        detection = Detection(300, 180, 100, 100, 0.9, "face")

        command = compute_command(detection, 416, 416, FACE_PROFILE)

        self.assertEqual(command.yaw, 27)
        self.assertEqual(command.up_down, 5)

    def test_missing_detection_stops_all_motion(self):
        command = compute_command(None, 416, 416, FACE_PROFILE)

        self.assertEqual(
            (command.left_right, command.forward_backward, command.up_down, command.yaw),
            (0, 0, 0, 0),
        )

    def test_speed_is_clamped_and_area_controls_forward_motion(self):
        detection = Detection(1000, -1000, 400, 400, 0.9, "face")

        command = compute_command(detection, 416, 416, FACE_PROFILE)

        self.assertEqual(command.yaw, FACE_PROFILE.max_yaw_speed)
        self.assertEqual(command.up_down, FACE_PROFILE.max_vertical_speed)
        self.assertEqual(command.forward_backward, -30)

    def test_intermediate_target_area_preserves_neutral_band(self):
        detection = Detection(208, 208, 48, 48, 0.9, "face")

        command = compute_command(detection, 416, 416, FACE_PROFILE)

        self.assertEqual(command.forward_backward, 0)


if __name__ == "__main__":
    unittest.main()
