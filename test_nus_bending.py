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
             patch("nus_bending.cv2.line", wraps=cv2.line) as lines:
            overlay.draw(frame.copy(), frame, None, None)
        self.assertTrue(lines.called)
        self.assertIn("T", labels)
        self.assertTrue(any(label.endswith(" deg") and label.startswith(("+", "-"))
                            for label in labels))

    def test_live_overlay_shows_only_two_largest_angle_sections(self):
        frame, settings = self.frame_and_settings()
        settings["reference_points"], _ = marker_chain(frame, settings)
        labels = []
        real_put_text = cv2.putText
        def capture_text(image, label, *args, **kwargs):
            labels.append(label)
            return real_put_text(image, label, *args, **kwargs)
        sections = [(1, 2, 12.0), (3, 4, -30.0), (6, 7, 18.0)]
        with patch("nus_bending.load_settings", return_value=settings), \
             patch("nus_bending.bend_sections", return_value=sections), \
             patch("nus_bending.cv2.putText", side_effect=capture_text):
            BendingOverlay(True).draw(frame.copy(), frame, None, None)
        angle_labels = {label for label in labels if label.endswith(" deg")
                        and label.startswith(("+", "-"))}
        self.assertEqual(angle_labels, {"-30.0 deg", "+18.0 deg"})
        summary = next(label for label in labels if label.startswith("Bending:"))
        self.assertIn("-30.0 deg", summary)
        self.assertIn("+18.0 deg", summary)
        self.assertNotIn("12.0 deg", summary)

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
        overlay.settings = settings
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

    def test_one_disc_in_two_windows_keeps_both_marker_identities(self):
        frame = np.full((240, 650, 3), 110, np.uint8)
        blue, green = (200, 150, 60), (90, 230, 90)
        previous = [(50, 120), (100, 120), (122, 120)] + [
            (x, 120) for x in (180, 235, 290, 345, 400, 455, 510)]
        cv2.circle(frame, previous[0], 10, green, -1)
        cv2.circle(frame, (111, 120), 15, blue, -1)  # One disc between B1 and B2.
        for point in previous[3:]:
            cv2.circle(frame, point, 10, blue, -1)
        hsv = lambda color: [int(v) for v in cv2.cvtColor(
            np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0]]
        settings = {"roi": [25, 80, 540, 90], "blue_hsv": hsv(blue),
                    "green_hsv": hsv(green)}
        found = follow_markers(frame, settings, previous, 0)
        self.assertEqual(sum(point is not None for point in found), 10)
        self.assertLess(np.linalg.norm(np.subtract(found[1], found[2])), 18)

        overlay = BendingOverlay(True)
        overlay.settings = {**settings, "base": [600, 120]}
        overlay.last_points = np.asarray(previous, float)
        overlay.last_good_points = overlay.last_points.copy()
        overlay.missed = np.zeros(10, int)
        points, status = overlay._stable_indexed(found)
        self.assertIn("2 overlapped", status)
        self.assertTrue(overlay.overlapped[1] and overlay.overlapped[2])
        self.assertEqual(int(np.count_nonzero(overlay.missed)), 0)
        self.assertLess(np.linalg.norm(points[1]-points[2]), 18)
        self.assertTrue(np.array_equal(overlay.last_good_points, previous))

    def test_overlapping_blue_tips_stay_drawn_until_they_separate(self):
        frame = np.full((240, 650, 3), 110, np.uint8)
        blue, green = (200, 150, 60), (90, 230, 90)
        previous = [(50, 120), (100, 120), (122, 120)] + [
            (x, 120) for x in (180, 235, 290, 345, 400, 455, 510)]
        cv2.circle(frame, previous[0], 10, green, -1)
        cv2.circle(frame, (111, 120), 15, blue, -1)
        for point in previous[3:]:
            cv2.circle(frame, point, 10, blue, -1)
        hsv = lambda color: [int(v) for v in cv2.cvtColor(
            np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0]]
        settings = {"frame_size": [650, 240], "roi": [25, 80, 540, 90],
                    "base": [600, 120], "blue_hsv": hsv(blue),
                    "green_hsv": hsv(green), "reference_points": previous}
        overlay = BendingOverlay(True)
        labels = []
        real_put_text = cv2.putText
        def capture_text(image, label, *args, **kwargs):
            labels.append(label)
            return real_put_text(image, label, *args, **kwargs)
        with patch("nus_bending.load_settings", return_value=settings), \
             patch("nus_bending.cv2.putText", side_effect=capture_text):
            for _ in range(6):
                overlay.draw(frame.copy(), frame)
        self.assertTrue(overlay.overlapped[1] and overlay.overlapped[2])
        self.assertTrue(np.all(overlay.missed == 0))
        self.assertIn("B1", labels)
        self.assertIn("B2", labels)
        self.assertTrue(any("angle paused until markers separate" in label for label in labels))
        separated = frame.copy()
        cv2.circle(separated, (111, 120), 17, (110, 110, 110), -1)
        cv2.circle(separated, previous[1], 9, blue, -1)
        cv2.circle(separated, previous[2], 9, blue, -1)
        for _ in range(4):
            overlay.draw(separated.copy(), separated)
        self.assertFalse(np.any(overlay.overlapped))
        self.assertTrue(np.all(overlay.missed == 0))

    def test_coincident_blue_assignments_hold_both_without_angle_error(self):
        previous = np.asarray([(50+i*22, 120) for i in range(10)], float)
        overlay = BendingOverlay(True)
        overlay.settings = {"base": [300, 120]}
        overlay.last_points = previous.copy()
        overlay.last_good_points = previous.copy()
        overlay.missed = np.zeros(10, int)
        duplicate = previous.copy()
        duplicate[2] = duplicate[1]
        points, status = overlay._stable_indexed(duplicate)
        self.assertIsNotNone(points)
        self.assertIn("2 overlapped", status)
        self.assertEqual(overlay.missed[2], 0)
        self.assertTrue(overlay.overlapped[1] and overlay.overlapped[2])
        self.assertTrue(np.array_equal(overlay.last_good_points, previous))

    def test_visible_markers_remain_tracked_during_long_single_marker_loss(self):
        frame, settings = self.frame_and_settings()
        reference, _ = marker_chain(frame, settings)
        settings["reference_points"] = reference
        for missing_index in (0, 4):
            with self.subTest(missing_index=missing_index):
                missing_frame = frame.copy()
                cv2.circle(missing_frame, tuple(map(round, reference[missing_index])),
                           18, (110, 110, 110), -1)
                overlay = BendingOverlay(True)
                with patch("nus_bending.load_settings", return_value=settings):
                    overlay.draw(frame.copy(), frame)
                    for _ in range(9):
                        overlay.draw(missing_frame.copy(), missing_frame)
                    self.assertIsNotNone(overlay.last_points)
                    self.assertEqual(int(np.count_nonzero(overlay.missed == 0)), 9)
                    overlay.draw(frame.copy(), frame)
                self.assertTrue(np.all(overlay.missed == 0))

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

    def test_auto_reacquires_shifted_return_pose_without_global_search(self):
        frame, settings = self.frame_and_settings()
        reference, _ = marker_chain(frame, settings)
        settings["reference_points"] = reference
        shifted = cv2.warpAffine(frame, np.float32([[1, 0, 20], [0, 1, 35]]),
                                 (frame.shape[1], frame.shape[0]), borderValue=(110, 110, 110))
        overlay = BendingOverlay(True)
        overlay.settings = settings
        overlay.last_good_points = np.asarray(reference, float)
        overlay.last_points = None  # A large bend previously lost the chain.
        with patch("nus_bending.marker_chain", side_effect=AssertionError("global search should not be needed")):
            points, status = overlay._recover(shifted)
        self.assertEqual(status, "10/10 reacquired")
        self.assertLess(np.max(np.linalg.norm(points-(np.asarray(reference)+[20, 35]), axis=1)), 4)

    def test_recovery_rejects_background_chain_with_wrong_spacing(self):
        frame, settings = self.frame_and_settings()
        reference, _ = marker_chain(frame, settings)
        settings["reference_points"] = reference
        overlay = BendingOverlay(True)
        overlay.settings = settings
        wrong = np.asarray(reference, float)
        wrong[5:, 0] += 160
        self.assertFalse(overlay._plausible_chain(wrong, reference))
        wrong = np.asarray(reference, float)
        wrong[5:, 1] += 75
        self.assertFalse(overlay._plausible_chain(wrong, reference))


if __name__ == "__main__":
    unittest.main()
