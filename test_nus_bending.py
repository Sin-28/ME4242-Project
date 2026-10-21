import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from nus_bending import (BendingOverlay, load_settings, marker_chain, save_settings,
                         spatial_angles, split_two_bends, turning_angles)


class BendingTests(unittest.TestCase):
    def frame_and_settings(self):
        frame = np.full((500, 1100, 3), 110, np.uint8)
        points = [(1040, 250)]
        for slope in (0, -.1, -.2, -.3, -.4, -.3, -.2, -.1, 0, .1):
            x, y = points[-1]
            points.append((x-100, y-100*slope))
        blue, green = (170, 60, 35), (40, 125, 55)
        for point in points[1:-1]:
            cv2.circle(frame, tuple(map(round, point)), 15, blue, -1)
        cv2.circle(frame, tuple(map(round, points[-1])), 15, green, -1)
        hsv = lambda color: [int(v) for v in cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0]]
        settings = {"frame_size": [1100, 500], "roi": [20, 180, 950, 280],
                    "base": list(points[0]), "blue_hsv": hsv(blue), "green_hsv": hsv(green)}
        return frame, settings

    def test_calibration_requires_matching_frame(self):
        _, settings = self.frame_and_settings()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bending.json"
            save_settings((500, 1100, 3), settings["roi"], settings["base"],
                          settings["blue_hsv"], settings["green_hsv"], path)
            self.assertEqual(load_settings((500, 1100, 3), path), settings)
            self.assertIsNone(load_settings((480, 640, 3), path))

    def test_two_bends_and_depth_from_existing_frames(self):
        frame, settings = self.frame_and_settings()
        chain, status = marker_chain(frame, settings)
        self.assertEqual(len(chain), 10)
        ordered = [settings["base"]] + chain[::-1]
        bend = split_two_bends(turning_angles(ordered))
        self.assertIsNotNone(bend)
        self.assertLess(bend[1], -15)
        self.assertGreater(bend[2], 15)
        depth = np.full(frame.shape[:2], 1000, np.uint16)
        for point in ordered[len(ordered)//2:]:
            cv2.circle(depth, tuple(map(int, point)), 16, 1200, -1)
        values, message = spatial_angles(ordered, bend, depth,
                                         lambda x, y, z: ((x-550)*z/500, (y-250)*z/500, z))
        self.assertEqual(len(values), 2)
        self.assertTrue(all(np.isfinite(values)))
        self.assertIn("3D base", message)
        depth[:] = 0
        values, message = spatial_angles(ordered, bend, depth,
                                         lambda x, y, z: ((x-550)*z/500, (y-250)*z/500, z))
        self.assertIsNone(values)
        self.assertIn("showing 2D", message)

    def test_disabled_overlay_is_inert(self):
        frame, _ = self.frame_and_settings()
        output = frame.copy()
        overlay = BendingOverlay(enabled=False)
        with patch("nus_bending.marker_chain", side_effect=AssertionError("must stay off")):
            self.assertIs(overlay.draw(output, frame, None, None), output)
        self.assertTrue(np.array_equal(output, frame))


if __name__ == "__main__":
    unittest.main()
