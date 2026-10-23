import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from nus_bending import (BendingOverlay, add_manual_marker, bend_sections, follow_markers, load_settings, marker_chain,
                         save_settings, spatial_angles, spatial_points, split_two_bends, turning_angles)


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

    def test_depth_tracking_requires_saved_depth_and_aligned_frame(self):
        frame, settings = self.frame_and_settings()
        reference, _ = marker_chain(frame, settings)
        settings["reference_points"] = reference
        overlay = BendingOverlay(True, depth_tracking=True)
        with patch("nus_bending.load_settings", return_value=settings), \
             patch("nus_bending.follow_markers", side_effect=AssertionError("must require depth setup")):
            overlay.draw(frame.copy(), frame, np.full(frame.shape[:2], 1000, np.float32), None)
        settings["reference_depth_mm"] = [1000.0]*10
        overlay.shape = None
        with patch("nus_bending.load_settings", return_value=settings), \
             patch("nus_bending.follow_markers", side_effect=AssertionError("must require aligned depth")):
            overlay.draw(frame.copy(), frame, np.zeros((240, 650), np.float32), None)

    def test_depth_rejects_same_color_background_inside_tracking_window(self):
        frame = np.full((240, 650, 3), 110, np.uint8)
        depth = np.full((240, 650), 1250, np.float32)
        previous = [(50, 120)] + [(x, 120) for x in (100, 155, 210, 265, 320, 375, 430, 485, 540)]
        blue, green = (200, 150, 60), (90, 230, 90)
        for index, point in enumerate(previous):
            color = green if index == 0 else blue
            cv2.circle(frame, point, 7, color, -1)
            cv2.circle(depth, point, 7, 1000, -1)
        # A false blue disc lies exactly at B4's old center. The true B4 moved
        # within the local search window but remains at the rod depth.
        old = previous[4]
        cv2.circle(depth, old, 7, 1250, -1)
        cv2.circle(frame, (old[0], old[1]+17), 7, blue, -1)
        cv2.circle(depth, (old[0], old[1]+17), 7, 1000, -1)
        hsv = lambda color: [int(v) for v in cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0]]
        settings = {"roi": [25, 80, 550, 90], "blue_hsv": hsv(blue),
                    "green_hsv": hsv(green)}
        found = follow_markers(frame, settings, previous, 0, depth, [1000]*10)
        self.assertIsNotNone(found[4])
        self.assertGreater(found[4][1], old[1]+12)
        depth[:] = 0
        self.assertTrue(all(point is None for point in
                            follow_markers(frame, settings, previous, 0, depth, [1000]*10)))

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

    def test_three_bends_and_out_of_plane_depth(self):
        headings = np.radians([0, 0, 15, 30, 15, 0, 0, 15, 30, 30])
        xy = [(0., 0.)]
        for angle in headings:
            xy.append(tuple(np.asarray(xy[-1]) + 25*np.array([np.cos(angle), np.sin(angle)])))
        self.assertEqual(len(bend_sections(xy)), 3)
        xyz = [(float(i*25), 0., float(1200 + 90*np.sin(i*np.pi/5))) for i in range(11)]
        self.assertGreaterEqual(len(bend_sections(xyz, min_bend=8)), 1)
        depth = np.full((100, 350), 1200, np.uint16)
        points = [(i*25+20, 50) for i in range(11)]
        for i, point in enumerate(points):
            cv2.circle(depth, point, 8, int(xyz[i][2]), -1)
        recovered, message = spatial_points(points, depth,
                                            lambda x, y, z: (x-20, y-50, z))
        self.assertIn("3D", message)
        self.assertGreaterEqual(len(bend_sections(recovered, min_bend=8)), 1)


if __name__ == "__main__":
    unittest.main()
