"""Optional, read-only bending overlay for the existing EMIO color/depth frames."""

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).with_name("bending_config.json")
BLUE_HSV = (112, 120, 80)
GREEN_HSV = (101, 70, 75)
MARKER_COUNT = 10  # nine blue discs and one green tip


def load_settings(shape, path=CONFIG_PATH):
    try:
        settings = json.loads(Path(path).read_text())
        height, width = shape[:2]
        roi, base = settings["roi"], settings["base"]
        if settings["frame_size"] != [width, height]:
            return None
        if not (len(roi) == 4 and 0 <= roi[0] < width and 0 <= roi[1] < height
                and roi[2] > 0 and roi[3] > 0 and roi[0]+roi[2] <= width
                and roi[1]+roi[3] <= height):
            return None
        if not (len(base) == 2 and 0 <= base[0] < width and 0 <= base[1] < height):
            return None
        for key in ("blue_hsv", "green_hsv"):
            values = settings[key]
            if len(values) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in values):
                return None
        reference = settings.get("reference_points")
        if reference is not None and (len(reference) != MARKER_COUNT or
                                      any(len(point) != 2 or not (0 <= point[0] < width and
                                                                   0 <= point[1] < height)
                                          for point in reference)):
            return None
        return settings
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return None


def save_settings(shape, roi, base, blue_hsv, green_hsv, path=CONFIG_PATH, reference_points=None):
    height, width = shape[:2]
    data = {"frame_size": [width, height], "roi": list(roi), "base": list(base),
            "blue_hsv": list(blue_hsv), "green_hsv": list(green_hsv)}
    if reference_points is not None:
        data["reference_points"] = [list(point) for point in reference_points]
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def color_mask(frame, target, tolerance, roi):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = target
    distance = abs(hsv[..., 0].astype(np.int16) - hue)
    distance = np.minimum(distance, 180 - distance)
    mask = ((distance <= tolerance) & (hsv[..., 1] >= max(35, int(sat * .45)))
            & (hsv[..., 2] >= max(20, int(val * .12)))).astype(np.uint8) * 255
    x, y, w, h = roi
    outside = np.zeros_like(mask)
    outside[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
    return cv2.morphologyEx(outside, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def tracking_roi(settings, shape):
    """Keep the base-side discs in view even if setup ROI ended too early."""
    height, width = shape[:2]
    x, y, w, h = settings["roi"]
    base_x, base_y = settings.get("base", (x+w, y+h//2))
    x0 = max(0, x-50)
    y0 = max(0, min(y-60, base_y-100))
    x1 = min(width, max(x+w, base_x+30))
    y1 = min(height, max(y+h+60, base_y+100))
    return x0, y0, x1-x0, y1-y0


def marker_chain(frame, settings, previous=None, hue_offset=0):
    """Locate colored discs along image x; returns left-to-right centers."""
    roi = tracking_roi(settings, frame.shape)
    x0, y0, width, height = roi
    blue_target = ((settings["blue_hsv"][0] + hue_offset) % 180, *settings["blue_hsv"][1:])
    green_target = ((settings["green_hsv"][0] + hue_offset) % 180, *settings["green_hsv"][1:])
    blue_hue, green_hue = blue_target[0], green_target[0]
    hue_gap = min(abs(blue_hue-green_hue), 180-abs(blue_hue-green_hue))
    distinct_tip = hue_gap >= 16
    blue_mask = color_mask(frame, blue_target,
                           min(15, hue_gap-6) if distinct_tip else 15, roi)
    green_mask = color_mask(frame, green_target, 7, roi)
    mask = blue_mask if distinct_tip else cv2.bitwise_or(blue_mask, green_mask)
    region = mask[y0:y0+height, x0:x0+width]
    projection = (region > 0).sum(axis=0).astype(np.float32)
    projection = cv2.GaussianBlur(projection.reshape(1, -1), (0, 0), 7).ravel()
    threshold = max(12.0, .18 * float(projection.max()))
    candidates = [(float(projection[x]), x) for x in range(1, width-1)
                  if projection[x] > threshold and projection[x] >= projection[x-1]
                  and projection[x] > projection[x+1]]
    separation = max(18, round(width / (MARKER_COUNT+1) * .38))
    peaks = []
    for _strength, x in sorted(candidates, reverse=True):
        if all(abs(x-other) >= separation for other in peaks):
            peaks.append(x)
    peak_limit = MARKER_COUNT-1 if distinct_tip else MARKER_COUNT
    if len(peaks) > peak_limit:
        from itertools import combinations
        strengths = {x: float(projection[x]) for x in peaks}
        def regularity(selected):
            gaps = np.diff(sorted(selected)).astype(float)
            return float(np.sum(np.diff(np.log(gaps))**2)) - .001 * sum(strengths[x] for x in selected)
        peaks = list(min(combinations(peaks, peak_limit), key=regularity))
    if len(peaks) < 7:
        return [], f"markers {len(peaks)}/{MARKER_COUNT}"
    points = []
    for x in sorted(peaks):
        half = max(8, separation // 3)
        ys, _ = np.nonzero(region[:, max(0, x-half):min(width, x+half+1)])
        if len(ys) < 80:
            return [], "too few colored pixels"
        points.append((float(x0+x), float(y0+np.median(ys))))
    for i in range(len(points)-2, 0, -1):
        left, mid, right = points[i-1:i+2]
        expected = left[1] + (right[1]-left[1])*(mid[0]-left[0])/(right[0]-left[0])
        if abs(mid[1]-expected) > 80:
            points.pop(i)
    if distinct_tip and points:
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tips = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area < 100 or area > 3000 or w < 8 or h < 8 or max(w/h, h/w) > 2.5:
                continue
            moments = cv2.moments(contour)
            if moments["m00"]:
                center = (moments["m10"]/moments["m00"], moments["m01"]/moments["m00"])
                gap = points[0][0]-center[0]
                expected_y = points[0][1]
                if len(points) > 1:
                    expected_y += (points[0][1]-points[1][1])*gap/(points[1][0]-points[0][0])
                if 7 < gap < 110 and abs(center[1]-expected_y) < 55:
                    tips.append((area, center))
        if not tips:
            return [], "green tip not found"
        if previous is not None:
            tips.sort(key=lambda item: np.linalg.norm(np.subtract(item[1], previous[0])))
        else:
            tips.sort(key=lambda item: (abs((points[0][0]-item[1][0]) - 25), -item[0]))
        points.insert(0, tips[0][1])
    if len(points) < 7:
        return [], f"markers {len(points)}/{MARKER_COUNT}"
    return points, f"markers {len(points)}/{MARKER_COUNT}"


def follow_markers(frame, settings, previous, hue_offset, missed=None):
    """Find each known marker in a small window around its last location."""
    blue_target = ((settings["blue_hsv"][0]+hue_offset) % 180, *settings["blue_hsv"][1:])
    green_target = ((settings["green_hsv"][0]+hue_offset) % 180, *settings["green_hsv"][1:])
    roi = tracking_roi(settings, frame.shape)
    blue_mask = color_mask(frame, blue_target, 13, roi)
    green_mask = color_mask(frame, green_target, 9, roi)
    height, width = frame.shape[:2]
    found = []
    for index, (x, y) in enumerate(previous):
        x, y = int(round(x)), int(round(y))
        neighbors = [abs(x-previous[j][0]) for j in (index-1, index+1)
                     if 0 <= j < len(previous)]
        lost = int(missed[index]) if missed is not None else 0
        radius_x = (min(50, 22+lost*8) if index == 0 else
                    min(32, max(12, int(min(neighbors)*.45)+lost*5)))
        radius_y = min(90, 27+lost*12)
        x0, x1 = max(0, x-radius_x), min(width, x+radius_x+1)
        y0, y1 = max(0, y-radius_y), min(height, y+radius_y+1)
        mask = green_mask if index == 0 else blue_mask
        region = mask[y0:y1, x0:x1]
        count, labels, stats, centers = cv2.connectedComponentsWithStats(region)
        candidates = []
        for component in range(1, count):
            area = stats[component, cv2.CC_STAT_AREA]
            if area < (18 if index == 0 else 45):
                continue
            center = centers[component] + (x0, y0)
            candidates.append((float(np.linalg.norm(center-(x, y))), tuple(center)))
        found.append(min(candidates)[1] if candidates else None)
    return found


def turning_angles(points):
    vectors = np.diff(np.asarray(points, float), axis=0)
    if len(vectors) < 2 or np.any(np.linalg.norm(vectors, axis=1) < 3):
        raise ValueError("markers overlap")
    a, b = vectors[:-1], vectors[1:]
    return np.degrees(np.arctan2(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0],
                                 np.sum(a*b, axis=1)))


def valid_chain(points, base):
    """Reject duplicated detections before they can poison the tracker."""
    if points is None or len(points) != MARKER_COUNT:
        return False
    ordered = np.asarray(points, float)
    gaps = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    return bool(np.all(np.isfinite(ordered)) and np.all(gaps >= 7))


def effective_base(settings, points):
    """Extrapolate the base when its saved pixel falls behind the last disc."""
    base = np.asarray(settings["base"], float)
    if base[0] > points[-1][0]+7:
        return tuple(base)
    reference = settings.get("reference_points")
    gap = (base[0]-reference[-1][0]) if reference is not None else 75
    gap = max(30., float(gap))
    tangent = np.asarray(points[-1], float)-np.asarray(points[-2], float)
    tangent /= max(np.linalg.norm(tangent), 1e-9)
    return tuple(np.asarray(points[-1])+gap*tangent)


def split_two_bends(turns, min_bend=15.0):
    best = None
    for cut in range(2, len(turns)-1):
        left, right = float(turns[:cut].sum()), float(turns[cut:].sum())
        if left*right >= 0 or min(abs(left), abs(right)) < min_bend:
            continue
        coherence = (abs(left)+abs(right))/(float(np.abs(turns).sum())+1e-9)
        if coherence < .65:
            continue
        score = min(abs(left), abs(right))*coherence
        if best is None or score > best[0]:
            best = (score, cut, left, right)
    return None if best is None else best[1:]


def bend_sections(points, min_bend=15.0):
    """Return independent bends as (first marker, last marker, angle).

    Adjacent turns in the same direction belong to one bend. A reversal starts
    another section; small color/depth jitter is ignored.
    """
    positions = np.asarray(points, float)
    # Central chords suppress per-marker pixel jitter while preserving bends.
    vectors = positions[2:]-positions[:-2]
    lengths = np.linalg.norm(vectors, axis=1)
    if len(vectors) < 2 or np.any(lengths < 3):
        return []
    unit = vectors / lengths[:, None]
    if unit.shape[1] == 2:
        turns = np.degrees(np.arctan2(unit[:-1, 0]*unit[1:, 1]-unit[:-1, 1]*unit[1:, 0],
                                     np.sum(unit[:-1]*unit[1:], axis=1)))
        directions = np.sign(turns)
        magnitudes = np.abs(turns)
    else:
        crosses = np.cross(unit[:-1], unit[1:])
        magnitudes = np.degrees(np.arctan2(np.linalg.norm(crosses, axis=1),
                                           np.sum(unit[:-1]*unit[1:], axis=1)))
        directions = crosses / np.maximum(np.linalg.norm(crosses, axis=1)[:, None], 1e-9)
    groups = []
    for index, magnitude in enumerate(magnitudes):
        if magnitude < 3.5:
            continue
        direction = directions[index]
        same = bool(groups) and (directions[groups[-1][-1]]*direction > 0
                                 if unit.shape[1] == 2 else
                                 np.dot(directions[groups[-1][-1]], direction) > .35)
        if same and index-groups[-1][-1] <= 2:
            groups[-1].append(index)
        else:
            groups.append([index])
    sections = []
    for group in groups:
        first, last = group[0], group[-1]+1
        if unit.shape[1] == 2:
            a, b = unit[first], unit[last]
            angle = math.degrees(math.atan2(a[0]*b[1]-a[1]*b[0], float(np.dot(a, b))))
        else:
            angle = angle_between(unit[first], unit[last])
        if abs(angle) >= min_bend:
            sections.append((first, last+2, float(angle)))
    return sections


def spatial_points(points, depth, pixel_to_robot):
    if depth is None or depth.ndim != 2 or pixel_to_robot is None:
        return None, "depth unavailable"
    xyz = []
    for x, y in points:
        value = depth_at(depth, (x, y))
        xyz.append(None if value is None else np.asarray(pixel_to_robot(x, y, value), float))
    valid = sum(point is not None and np.all(np.isfinite(point)) for point in xyz)
    if valid != len(points):
        return None, f"depth {valid}/{len(points)}; showing 2D"
    return np.asarray(xyz), "3D depth ready"


def depth_at(depth, point):
    x, y = (int(round(v)) for v in point)
    height, width = depth.shape
    for radius in (4, 8, 14):
        x0, x1 = max(0, x-radius), min(width, x+radius+1)
        y0, y1 = max(0, y-radius), min(height, y+radius+1)
        if x0 >= x1 or y0 >= y1:
            return None
        yy, xx = np.ogrid[y0:y1, x0:x1]
        values = depth[y0:y1, x0:x1][(xx-x)**2+(yy-y)**2 <= radius**2]
        values = values[(values > 0) & (values < 2000)]  # emioapi supplies millimeters
        if len(values) >= 5:
            return float(np.median(values))
    return None


def angle_between(a, b):
    return math.degrees(math.acos(np.clip(float(np.dot(a, b) /
                                               (np.linalg.norm(a)*np.linalg.norm(b))), -1, 1)))


def spatial_angles(points, bend, depth, pixel_to_robot):
    if depth is None or depth.ndim != 2:
        return None, "depth unavailable"
    xyz = []
    for x, y in points:
        value = depth_at(depth, (x, y))
        xyz.append(None if value is None else np.asarray(pixel_to_robot(x, y, value), float))
    valid = sum(point is not None and np.all(np.isfinite(point)) for point in xyz)
    if valid != len(points):
        return None, f"depth {valid}/{len(points)}; showing 2D"
    xyz = np.asarray(xyz)
    def section(start, end):
        if end-start < 2:
            return None
        first, last = xyz[start+2]-xyz[start], xyz[end]-xyz[end-2]
        if min(np.linalg.norm(first), np.linalg.norm(last)) < 1e-5:
            return None
        return angle_between(first, last)
    if bend is None:
        value = section(0, len(points)-1)
        return (value,), f"3D {value:.1f} deg" if value is not None else "depth geometry invalid"
    cut = bend[0]
    values = (section(0, cut+1), section(cut, len(points)-1))
    return values, (f"3D base {values[0]:.1f} | tip {values[1]:.1f} deg"
                    if None not in values else "depth geometry invalid")


class BendingOverlay:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.settings = None
        self.shape = None
        self.last_points = None
        self.missed = None
        self.hue_offset = 0
        self.prev_gray = None

    def _recover(self, frame):
        """Retry the saved straight pose and the original hue after tracking loss."""
        reference = self.settings.get("reference_points")
        if reference is not None:
            previous = np.asarray(reference, float)
            for offset in (self.hue_offset, 0, 10, -10, 20, -20):
                found = follow_markers(frame, self.settings, previous, offset)
                if all(point is not None for point in found) and valid_chain(found, self.settings["base"]):
                    self.last_points = np.asarray(found, float)
                    self.missed = np.zeros(MARKER_COUNT, int)
                    self.hue_offset = offset
                    return self.last_points.copy(), "10/10 reacquired"
        best, status = [], "markers 0/10"
        for offset in dict.fromkeys((self.hue_offset, 0, 2, -2, 4, -4, 5, -5,
                                         10, -10, 15, -15, 20, -20)):
            candidate, candidate_status = marker_chain(frame, self.settings, None, offset)
            if len(candidate) > len(best):
                best, status = candidate, candidate_status
            if valid_chain(candidate, self.settings["base"]):
                self.last_points = np.asarray(candidate, float)
                self.missed = np.zeros(MARKER_COUNT, int)
                self.hue_offset = offset
                return self.last_points.copy(), "10/10 reacquired"
        return None, status

    def _current_hue_offset(self, frame):
        """Sample known blue discs so camera auto exposure cannot stale the saved hue."""
        if self.last_points is None:
            return self.hue_offset
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        hues = []
        for x, y in self.last_points[1:]:
            x, y = int(round(x)), int(round(y))
            patch = hsv[max(0, y-4):min(height, y+5), max(0, x-4):min(width, x+5)]
            pixels = patch[(patch[..., 1] >= 55) & (patch[..., 2] >= 20)]
            if len(pixels) >= 8:
                hues.append(float(np.median(pixels[:, 0])))
        if len(hues) >= 4:
            self.hue_offset = int(np.clip(np.median(hues)-self.settings["blue_hsv"][0], -25, 25))
        return self.hue_offset

    def _stable_points(self, detected):
        if self.last_points is None:
            if len(detected) != MARKER_COUNT:
                return None, f"{len(detected)}/{MARKER_COUNT}; waiting for full chain"
            self.last_points = np.asarray(detected, float)
            self.missed = np.zeros(MARKER_COUNT, int)
            return self.last_points.copy(), "10/10"
        updated = self.last_points.copy()
        used = set()
        for point in detected:
            distances = np.linalg.norm(self.last_points-np.asarray(point), axis=1)
            choices = [(distances[i], i) for i in range(MARKER_COUNT) if i not in used]
            if not choices:
                break
            distance, index = min(choices)
            if distance <= 50:
                updated[index] = .65*self.last_points[index] + .35*np.asarray(point)
                self.missed[index] = 0
                used.add(index)
        for index in range(MARKER_COUNT):
            if index not in used:
                self.missed[index] += 1
        if np.any(self.missed > 4):
            self.last_points = self.missed = None
            return None, "tracking lost; reacquiring markers"
        self.last_points = updated
        held = MARKER_COUNT-len(used)
        return updated.copy(), "10/10" if not held else f"{len(used)}/10 visible; {held} held"

    def _stable_indexed(self, detected):
        updated = self.last_points.copy()
        visible = 0
        for index, point in enumerate(detected):
            if point is not None and np.linalg.norm(self.last_points[index]-point) <= 100:
                updated[index] = .65*self.last_points[index]+.35*np.asarray(point)
                self.missed[index] = 0
                visible += 1
            else:
                self.missed[index] += 1
        if np.any(self.missed > 15) or not valid_chain(updated, self.settings["base"]):
            self.last_points = self.missed = None
            return None, "tracking lost; reacquiring markers"
        self.last_points = updated
        return updated.copy(), ("10/10" if visible == 10 else
                                f"{visible}/10 visible; {10-visible} held")

    def draw(self, image, frame, depth, vision):
        if not self.enabled:
            return image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_gray = self.prev_gray
        self.prev_gray = gray
        if self.shape != image.shape[:2]:
            self.shape = image.shape[:2]
            self.settings = load_settings(image.shape)
            self.last_points = self.missed = None
            self.hue_offset = 0
            previous_gray = None
            if self.settings is not None and self.settings.get("reference_points") is not None:
                self.last_points = np.asarray(self.settings["reference_points"], float)
                self.missed = np.zeros(MARKER_COUNT, int)
        if self.settings is None:
            self._text(image, "Bending: run `python nus_main.py bending-setup`", 0)
            return image
        offset = self._current_hue_offset(frame)
        if self.last_points is not None:
            predicted = self.last_points.copy()
            if previous_gray is not None and previous_gray.shape == gray.shape:
                flowed, status_flow, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray, gray, self.last_points.astype(np.float32).reshape(-1, 1, 2),
                    None, winSize=(31, 31), maxLevel=3)
                if flowed is not None:
                    flowed = flowed.reshape(-1, 2)
                    good = (status_flow.ravel() != 0) & (
                        np.linalg.norm(flowed-self.last_points, axis=1) < 120)
                    predicted[good] = flowed[good]
            detected = follow_markers(frame, self.settings, predicted, offset, self.missed)
            detection_status = f"local {sum(p is not None for p in detected)}/10"
            points, status = self._stable_indexed(detected)
            if points is None:
                points, status = self._recover(frame)
        else:
            points, status = self._recover(frame)
            detection_status = status
        if points is None:
            self._text(image, "Bending: " + status + " (" + detection_status + ")", 0)
            return image
        base = effective_base(self.settings, points)
        ordered = [base] + list(points[::-1])
        self._draw_chain(image, ordered)
        if self.missed is not None and np.any(self.missed):
            self._text(image, f"Bending: {status} | angle paused until markers return", 0)
            return image
        try:
            turns = turning_angles(ordered)
        except ValueError as exc:
            self._text(image, "Bending: tracking geometry invalid; reacquiring", 0)
            self.last_points = self.missed = None
            return image
        sections_2d = bend_sections(ordered)
        two_d = ("2D " + (" | ".join(f"B{i} {angle:+.1f} deg" for i, (_, _, angle)
                                    in enumerate(sections_2d, 1)) if sections_2d else "straight"))
        converter = vision.pixel_to_robot if vision is not None else None
        # The mount may be outside the image or have no usable depth. The ten
        # colored discs alone are sufficient to measure the rod's 3D bends.
        xyz, depth_status = spatial_points(points[::-1], depth, converter)
        if xyz is not None:
            sections_3d = bend_sections(xyz)
            depth_status = ("3D " + (" | ".join(f"B{i} {angle:.1f} deg" for i, (_, _, angle)
                                             in enumerate(sections_3d, 1)) if sections_3d else "straight"))
        for index, (start, end, angle) in enumerate(sections_2d, 1):
            center = tuple(np.rint((np.asarray(ordered[start])+ordered[end])/2).astype(int))
            self._label(image, f"B{index} {angle:+.0f}d", (center[0], center[1]-28))
        self._text(image, f"Bending: {status} | {two_d}", 0)
        self._text(image, depth_status, 1)
        return image

    @staticmethod
    def _label(image, label, position):
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 2)
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 255), 1)

    def _draw_chain(self, image, ordered):
        """Draw markers, path and tip directly at their measured image positions."""
        cv2.polylines(image, [np.rint(ordered).astype(np.int32)], False, (0, 0, 0), 3)
        cv2.polylines(image, [np.rint(ordered).astype(np.int32)], False, (0, 255, 255), 1)
        cv2.circle(image, tuple(map(int, ordered[0])), 9, (0, 0, 255), 1)
        for index, point in enumerate(ordered[1:], 1):
            center = tuple(np.rint(point).astype(int))
            color = (0, 255, 0) if index == MARKER_COUNT else (255, 100, 0)
            cv2.circle(image, center, 9, (0, 0, 0), 3)
            cv2.circle(image, center, 9, color, 1)
            self._label(image, "TIP" if index == MARKER_COUNT else str(index),
                        (center[0]+12, center[1]-12))

    @staticmethod
    def _text(image, text, row):
        y = image.shape[0]-48+row*21
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)


def sample_color(frame, point):
    x, y = point
    patch = frame[max(0, y-8):y+9, max(0, x-8):x+9]
    pixels = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    pixels = pixels[(pixels[:, 1] >= 35) & (pixels[:, 2] >= 20)]
    if len(pixels) < 10:
        raise ValueError("color not visible at click")
    return tuple(int(v) for v in np.median(pixels, axis=0))


def run_setup(camera):
    """One-time interactive calibration using the project's existing camera."""
    window = "Bending setup (r ROI, a base, b blue, g green, q quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    existing = None
    roi = base = None
    blue_hsv, green_hsv = BLUE_HSV, GREEN_HSV
    mode = None
    frame = None
    note = "r: ROI around all marker movement; a: fixed base; b/g: sample colors"

    def persist():
        nonlocal note
        if roi is not None and base is not None:
            try:
                current = {"roi": roi, "blue_hsv": blue_hsv, "green_hsv": green_hsv}
                reference, _ = marker_chain(frame, current)
                if len(reference) != MARKER_COUNT and existing is not None:
                    reference = existing.get("reference_points", [])
                save_settings(frame.shape, roi, base, blue_hsv, green_hsv,
                              reference_points=reference if len(reference) == MARKER_COUNT else None)
                note = ("Saved 10 marker positions; q to quit" if len(reference) == MARKER_COUNT
                        else "Saved colors/ROI; show all 10 markers and press s to save positions")
            except OSError as exc:
                note = f"Could not save calibration: {exc}"

    def mouse(event, x, y, _flags, _param):
        nonlocal mode, base, blue_hsv, green_hsv, note
        if event != cv2.EVENT_LBUTTONDOWN or frame is None or mode is None:
            return
        if mode == "a":
            base = (x, y)
            persist()
        elif mode in ("b", "g"):
            try:
                value = sample_color(frame, (x, y))
                if mode == "b": blue_hsv = value
                else: green_hsv = value
                persist()
            except ValueError as exc:
                note = str(exc)
        mode = None

    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            camera.update()
            frame = camera.frame.copy()
            if existing is None:
                existing = load_settings(frame.shape)
                if existing:
                    roi, base = tuple(existing["roi"]), tuple(existing["base"])
                    blue_hsv, green_hsv = tuple(existing["blue_hsv"]), tuple(existing["green_hsv"])
                    note = "Saved calibration loaded; choose r/a/b/g to change it"
            view = frame.copy()
            if roi: cv2.rectangle(view, roi[:2], (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 255), 2)
            if base: cv2.drawMarker(view, base, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            BendingOverlay._text(view, note, 0)
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), 27): break
            if key in (ord("a"), ord("b"), ord("g")):
                mode = chr(key)
                note = f"Click {'base' if mode == 'a' else 'blue marker' if mode == 'b' else 'green tip'}"
            if key == ord("s"):
                persist()
            if key == ord("r"):
                selected = cv2.selectROI(window, frame, False, False)
                if selected[2] and selected[3]:
                    roi = tuple(int(v) for v in selected)
                    persist()
                cv2.setMouseCallback(window, mouse)
    finally:
        cv2.destroyWindow(window)


class RGBCamera:
    """OpenCV color-only camera for setup/preview when the depth SDK is unavailable."""
    def __init__(self, index=0):
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open RGB camera {index}")
        self.frame = None

    def update(self):
        ok, self.frame = self.capture.read()
        if not ok or self.frame is None:
            raise RuntimeError("Could not read RGB frame")

    def close(self):
        self.capture.release()


class RealSenseBendingCamera:
    """Aligned color/depth stream for a motor-free 3D bending preview."""
    def __init__(self, width=1280, height=720):
        probe_code = ("import pyrealsense2 as rs; "
                      "p=rs.pipeline(); c=rs.config(); "
                      f"c.enable_stream(rs.stream.color,{width},{height},rs.format.bgr8,30); "
                      f"c.enable_stream(rs.stream.depth,{width},{height},rs.format.z16,30); "
                      "p.start(c); p.stop()")
        probe = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True, text=True, timeout=12)
        if probe.returncode != 0:
            raise RuntimeError(f"RealSense depth stream failed (exit {probe.returncode}); "
                               "3D bending cannot run on this computer. "
                               "Use bending-preview without --depth for RGB tracking.")
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.frame = self.depth_frame = None
        self.intrinsics = None

    def update(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("No synchronized color/depth frames")
        self.frame = np.asanyarray(color.get_data())
        self.depth_frame = np.rint(np.asanyarray(depth.get_data()).astype(float)
                                    * self.depth_scale * 1000).astype(np.uint16)
        self.intrinsics = color.profile.as_video_stream_profile().intrinsics

    def pixel_to_robot(self, x, y, depth_mm):
        return np.asarray(self.rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [float(x), float(y)], float(depth_mm)), float)

    def close(self):
        self.pipeline.stop()


def run_preview(camera):
    """Test marker tracking without the motor controller."""
    window = "Bending preview (v toggle, q quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    overlay = BendingOverlay(enabled=True)
    try:
        while True:
            camera.update()
            frame = camera.frame.copy()
            overlay.draw(frame, camera.frame, getattr(camera, "depth_frame", None),
                         camera if hasattr(camera, "pixel_to_robot") else None)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), 27):
                break
            if key == ord("v"):
                overlay.enabled = not overlay.enabled
    finally:
        cv2.destroyWindow(window)
