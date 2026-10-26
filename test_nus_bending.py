import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from nus_bending import (BendingOverlay, RGBCamera, add_manual_marker, bend_sections, follow_markers, load_settings, marker_chain,
                         save_settings, split_two_bends, turning_angles)


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

    def test_manual_tip_and_blue_clicks_save_ordered_reference(self):
        roi = [20, 80, 510, 90]
        points = add_manual_marker([], "g", (50, 120), roi)
        for index in range(1, 10):
            points = add_manual_marker(points, "b", (50+index*50, 120), roi)
        self.assertEqual(len(points), 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bending.json"
            save_settings((240, 650, 3), roi, [600, 120], [101, 150, 220],
                          [60, 150, 220], path, points)
            self.assertEqual(load_settings((240, 650, 3), path)["reference_points"],
                             [list(point) for point in points])
        with self.assertRaisesRegex(ValueError, "left to right"):
            add_manual_marker(points[:2], "b", (70, 120), roi)
        with self.assertRaisesRegex(ValueError, "inside the ROI"):
            add_manual_marker([], "g", (10, 120), roi)

    def test_calibration_requires_matching_frame(self):
        _, settings = self.frame_and_settings()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bending.json"
            save_settings((500, 1100, 3), settings["roi"], settings["base"],
                          settings["blue_hsv"], settings["green_hsv"], path)
            self.assertEqual(load_settings((500, 1100, 3), path), settings)
            self.assertIsNone(load_settings((480, 640, 3), path))

    def test_two_bends_in_2d(self):
        frame, settings = self.frame_and_settings()
        chain, status = marker_chain(frame, settings)
        self.assertEqual(len(chain), 10, status)
        ordered = [settings["base"]] + chain[::-1]
        bend = split_two_bends(turning_angles(ordered))
        self.assertIsNotNone(bend)
        self.assertLess(bend[1], -15)
        self.assertGreater(bend[2], 15)

    def test_three_2d_bends(self):
        headings = np.radians([0, 0, 15, 30, 15, 0, 0, 15, 30, 30])
        xy = [(0., 0.)]
        for angle in headings:
            xy.append(tuple(np.asarray(xy[-1]) + 25*np.array([np.cos(angle), np.sin(angle)])))
        self.assertEqual(len(bend_sections(xy)), 3)

    def test_live_overlay_draws_tip_line_and_local_angles(self):
        frame, settings = self.frame_and_settings()
        reference, _ = marker_chain(frame, settings)
        settings["reference_points"] = reference
        overlay = BendingOverlay(enabled=True)
        labels = []
        real_put_text = cv2.putText
        def capture_text(image, text, *args, **kwargs):
            labels.append(text)
            return real_put_text(image, text, *args, **kwargs)
        with patch("nus_bending.load_settings", return_value=settings), \
             patch("nus_bending.cv2.putText", side_effect=capture_text), \
             patch("nus_bending.cv2.polylines", wraps=cv2.polylines) as lines:
            overlay.draw(frame.copy(), frame, None, None)
        self.assertTrue(lines.called)
        self.assertIn("T", labels)
        self.assertTrue(any(label.endswith(" deg") and label.startswith(("+", "-"))
                            for label in labels))

    def test_camera_requests_saved_frame_size_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bending.json"
            path.write_text('{"frame_size": [1280, 720]}')
            with patch("nus_bending.CONFIG_PATH", path), \
                 patch("nus_bending.cv2.VideoCapture") as capture:
                capture.return_value.isOpened.return_value = True
                camera = RGBCamera()
                camera.close()
            capture.return_value.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.return_value.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def test_angle_pauses_while_saved_marker_is_held(self):
        frame, settings = self.frame_and_settings()
        reference, _ = marker_chain(frame, settings)
        settings["reference_points"] = reference
        blank = np.full_like(frame, 110)
        labels = []
        real_put_text = cv2.putText
        def capture_text(image, label, *args, **kwargs):
            labels.append(label)
            return real_put_text(image, label, *args, **kwargs)
        with patch("nus_bending.load_settings", return_value=settings), \
             patch("nus_bending.cv2.putText", side_effect=capture_text):
            BendingOverlay(True).draw(blank.copy(), blank, None, None)
        self.assertTrue(any("angle paused" in label for label in labels))
        self.assertFalse(any(label.endswith(" deg") for label in labels))

    def test_disabled_overlay_is_inert(self):
        frame, _ = self.frame_and_settings()
        output = frame.copy()
        overlay = BendingOverlay(enabled=False)
        with patch("nus_bending.marker_chain", side_effect=AssertionError("must stay off")):
            self.assertIs(overlay.draw(output, frame, None, None), output)
        self.assertTrue(np.array_equal(output, frame))

    def test_distinct_green_tip_next_to_blue_disc(self):
        frame = np.full((240, 650, 3), 110, np.uint8)
        blue, green = (200, 150, 60), (90, 230, 90)
        cv2.circle(frame, (50, 120), 10, green, -1)
        for x in (72, 125, 180, 235, 290, 345, 400, 455, 510):
            cv2.circle(frame, (x, 120), 10, blue, -1)
        hsv = lambda color: [int(v) for v in cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0]]
        points, status = marker_chain(frame, {"roi": [30, 90, 510, 60],
                                              "blue_hsv": hsv(blue), "green_hsv": hsv(green)})
        self.assertEqual(len(points), 10, status)
        self.assertAlmostEqual(points[0][0], 50, delta=3)
        self.assertAlmostEqual(points[1][0], 72, delta=3)

    def test_local_tracking_ignores_neighbor_and_adapts_to_darker_frame(self):
        height, width = 240, 650
        centers = [(50, 120)] + [(x, 120) for x in (72, 125, 180, 235, 290, 345, 400, 455, 510)]
        settings = {"roi": [30, 80, 510, 90], "base": [600, 120],
                    "blue_hsv": [101, 150, 220], "green_hsv": [60, 150, 220],
                    "reference_points": centers}
        hsv = np.zeros((height, width, 3), np.uint8)
        hsv[:] = (0, 0, 110)
        cv2.circle(hsv, centers[0], 9, (70, 145, 75), -1)
        for point in centers[1:]:
            cv2.circle(hsv, point, 9, (111, 145, 65), -1)
        cv2.circle(hsv, (50, 72), 9, (70, 145, 75), -1)  # moving object, not the tip
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        overlay = BendingOverlay(enabled=True)
        overlay.settings = settings
        overlay.last_points = np.asarray(centers, float)
        overlay.missed = np.zeros(10, int)
        self.assertEqual(overlay._current_hue_offset(frame), 10)
        detected = follow_markers(frame, settings, overlay.last_points, overlay.hue_offset)
        self.assertTrue(all(point is not None for point in detected))
        self.assertAlmostEqual(detected[0][0], centers[0][0], delta=2)
        self.assertAlmostEqual(detected[0][1], centers[0][1], delta=2)

    def test_stationary_markers_stay_at_same_positions_across_frames(self):
        frame, settings = self.frame_and_settings()
        reference, _ = marker_chain(frame, settings)
        overlay = BendingOverlay(True)
        overlay.last_points = np.asarray(reference, float)
        overlay.missed = np.zeros(10, int)
        for _ in range(15):
            detected = follow_markers(frame, settings, overlay.last_points, 0)
            self.assertTrue(all(point is not None for point in detected))
            updated, status = overlay._stable_indexed(detected)
            self.assertEqual(status, "10/10")
            self.assertLess(np.max(np.linalg.norm(updated-reference, axis=1)), 3)

    def test_background_discs_cannot_steal_stationary_markers(self):
        frame = np.full((360, 680, 3), 110, np.uint8)
        previous = [(50+i*55, 150) for i in range(10)]
        blue, green = (200, 150, 60), (90, 230, 90)
        for x, y in previous[1:]:
            cv2.circle(frame, (x, y), 10, blue, -1)
        cv2.circle(frame, previous[0], 10, green, -1)
        for index in (5, 6, 7, 8):
            x, y = previous[index]
            cv2.circle(frame, (x, y+75), 14, blue, -1)
        hsv = lambda color: [int(v) for v in cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0]]
        settings = {"roi": [25, 100, 560, 180], "blue_hsv": hsv(blue),
                    "green_hsv": hsv(green)}
        found = follow_markers(frame, settings, previous, 0)
        self.assertTrue(all(point is not None for point in found))
        for point, expected in zip(found, previous):
            self.assertLess(np.linalg.norm(np.subtract(point, expected)), 2)
        for index in (5, 6, 7, 8):
            x, y = previous[index]
            cv2.circle(frame, (x, y), 12, (110, 110, 110), -1)
        found = follow_markers(frame, settings, previous, 0)
        self.assertTrue(all(found[index] is None for index in (5, 6, 7, 8)))

    def test_reacquires_saved_pose_after_stale_hue_and_tracking_loss(self):
        frame, settings = self.frame_and_settings()
        reference, _ = marker_chain(frame, settings)
        settings["reference_points"] = reference
        overlay = BendingOverlay(True)
        overlay.settings = settings
        overlay.shape = frame.shape[:2]
        overlay.hue_offset = 24
        points, status = overlay._recover(frame)
        self.assertEqual(len(points), 10, status)
        self.assertIsNotNone(overlay.last_points)


if __name__ == "__main__":
    unittest.main()
