"""Optional, read-only 2D bending overlay for color camera frames."""

import json
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


def save_settings(shape, roi, base, blue_hsv, green_hsv, path=CONFIG_PATH,
                  reference_points=None):
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


def split_touching_blue_discs(components, stats, centroid, component, left, right):
    """Find two shape peaks inside a touching blue blob, if both are visible."""
    x, y, width, height, _ = stats[component]
    patch = np.pad((components[y:y+height, x:x+width] == component).astype(np.uint8), 1)
    distance = cv2.distanceTransform(patch, cv2.DIST_L2, 5)[1:-1, 1:-1]
    axis = np.asarray(right, float)-np.asarray(left, float)
    previous_gap = np.linalg.norm(axis)
    if previous_gap < 4:
        return None
    axis /= previous_gap
    yy, xx = np.indices(distance.shape)
    projection = (xx+x-centroid[0])*axis[0] + (yy+y-centroid[1])*axis[1]
    halves = (np.where(projection < 0, distance, 0),
              np.where(projection >= 0, distance, 0))
    peaks = []
    for half in halves:
        row, col = np.unravel_index(np.argmax(half), half.shape)
        strength = float(half[row, col])
        if strength < 4:
            return None
        peaks.append((np.array([x+col, y+row], float), strength))
    separation = np.linalg.norm(peaks[1][0]-peaks[0][0])
    if not (max(6, .3*previous_gap) <= separation <= 1.7*previous_gap):
        return None
    if min(peaks[0][1], peaks[1][1]) < .55*max(peaks[0][1], peaks[1][1]):
        return None
    path = np.linspace(peaks[0][0], peaks[1][0], max(3, int(separation)+1))
    columns = np.clip(np.rint(path[:, 0]-x).astype(int), 0, width-1)
    rows = np.clip(np.rint(path[:, 1]-y).astype(int), 0, height-1)
    if float(np.min(distance[rows, columns])) > .94*min(peaks[0][1], peaks[1][1]):
        return None
    return tuple(map(tuple, (peaks[0][0], peaks[1][0])))


def marker_chain(frame, settings, previous=None, hue_offset=0):
    """Locate colored discs along image x; returns left-to-right centers."""
    roi = settings["roi"]
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


def follow_markers(frame, settings, previous, hue_offset, masks=None):
    """Find each known marker in a small window around its last location."""
    blue_target = ((settings["blue_hsv"][0]+hue_offset) % 180, *settings["blue_hsv"][1:])
    green_target = ((settings["green_hsv"][0]+hue_offset) % 180, *settings["green_hsv"][1:])
    if masks is None:
        blue_mask = color_mask(frame, blue_target, 13, settings["roi"])
        green_mask = color_mask(frame, green_target, 9, settings["roi"])
    else:
        blue_mask, green_mask = masks
    _, components, stats, centroids = cv2.connectedComponentsWithStats(blue_mask)
    height, width = frame.shape[:2]
    found = []
    component_ids = []
    for index, (x, y) in enumerate(previous):
        x, y = int(round(x)), int(round(y))
        neighbors = [abs(x-previous[j][0]) for j in (index-1, index+1)
                     if 0 <= j < len(previous)]
        radius_x = min(20, max(10, int(min(neighbors)*.45)))
        radius_y = 27
        x0, x1 = max(0, x-radius_x), min(width, x+radius_x+1)
        y0, y1 = max(0, y-radius_y), min(height, y+radius_y+1)
        mask = green_mask if index == 0 else blue_mask
        local = mask[y0:y1, x0:x1]
        ys, xs = np.nonzero(local)
        if len(xs) < (18 if index == 0 else 45):
            found.append(None)
            component_ids.append(0)
        else:
            found.append((float(x0+np.median(xs)), float(y0+np.median(ys))))
            if index == 0:
                component_ids.append(0)
            else:
                ids = components[y0:y1, x0:x1][local != 0]
                component_ids.append(int(np.argmax(np.bincount(ids)[1:])+1))
    # Touching discs can form one color component while retaining two lobes.
    # Split only when its shape has two distinct distance-transform peaks.
    for index in range(2, len(found)):
        component = component_ids[index]
        if (component and component == component_ids[index-1]
                and stats[component, cv2.CC_STAT_AREA] <= 2500):
            peaks = split_touching_blue_discs(
                components, stats, centroids[component], component,
                previous[index-1], previous[index])
            if peaks is None:
                peaks = (tuple(map(float, centroids[component])),)*2
            found[index-1], found[index] = peaks
    return found


def turning_angles(points):
    vectors = np.diff(np.asarray(points, float), axis=0)
    if len(vectors) < 2 or np.any(np.linalg.norm(vectors, axis=1) < 3):
        raise ValueError("markers overlap")
    a, b = vectors[:-1], vectors[1:]
    return np.degrees(np.arctan2(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0],
                                 np.sum(a*b, axis=1)))


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
    another section; small color jitter is ignored.
    """
    vectors = np.diff(np.asarray(points, float), axis=0)
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
        if magnitude < 2.5:
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
        angle = float(np.sum(magnitudes[group]))
        if angle >= min_bend:
            sign = float(directions[group[0]]) if unit.shape[1] == 2 else 1.0
            sections.append((group[0], group[-1]+2, sign*angle))
    return sections


def estimate_hidden_blue(points, missed, base, reference):
    """Interpolate one occluded blue disc from its visible neighbors."""
    missing = np.flatnonzero(missed)
    if len(missing) != 1 or missing[0] == 0:
        return None
    index = int(missing[0])
    points = np.asarray(points, float).copy()
    reference = np.asarray(reference, float)
    left = points[index-1]
    right = points[index+1] if index+1 < MARKER_COUNT else np.asarray(base, float)
    ref_right = (reference[index+1] if index+1 < MARKER_COUNT else
                 np.asarray(base, float))
    ref_left = reference[index-1]
    span = np.linalg.norm(ref_right-ref_left)
    actual_span = np.linalg.norm(right-left)
    if span < 6 or actual_span < 6 or actual_span > 2.5*span:
        return None
    fraction = float(np.clip(np.linalg.norm(reference[index]-ref_left)/span, .15, .85))
    points[index] = left + fraction*(right-left)
    return points, index


class BendingOverlay:
    def __init__(self, enabled=False, config_path=CONFIG_PATH):
        self.enabled = enabled
        self.config_path = config_path
        self.settings = None
        self.shape = None
        self.last_points = None
        self.last_good_points = None
        self.missed = None
        self.overlapped = np.zeros(MARKER_COUNT, bool)
        self.hue_offset = 0
        self.recovery_ticks = 0

    def _plausible_chain(self, points, reference=None, overlap_pairs=None):
        """Reject a full color match that cannot be the ordered rod."""
        if len(points) != MARKER_COUNT:
            return False
        points = np.asarray(points, float)
        if overlap_pairs is None:
            overlap_pairs = np.zeros(MARKER_COUNT-1, bool)
        if (not np.all(np.isfinite(points)) or np.any((np.diff(points[:, 0]) <= 4) & ~overlap_pairs)
                or self.settings["base"][0] <= points[-1, 0]
                or np.linalg.norm(points[-1]-self.settings["base"]) < 6
                or np.any(np.abs(np.diff(points[:, 1])) > 80)):
            return False
        if reference is not None:
            gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
            saved_gaps = np.linalg.norm(np.diff(reference, axis=0), axis=1)
            if np.any((gaps < .45*saved_gaps) & ~overlap_pairs) or np.any(gaps > 2.1*saved_gaps):
                return False
            shift = np.median(points-reference, axis=0)
            residual = np.linalg.norm(points-reference-shift, axis=1)
            if np.quantile(residual, .8) > 35 or np.max(residual) > 70:
                return False
        return True

    def _accept_recovery(self, points, offset):
        self.last_points = np.asarray(points, float)
        self.last_good_points = self.last_points.copy()
        self.missed = np.zeros(MARKER_COUNT, int)
        self.overlapped[:] = False
        self.hue_offset = offset
        self.recovery_ticks = 0
        return self.last_points.copy(), "10/10 reacquired"

    def _accept_partial_recovery(self, found, reference, offset):
        points = np.asarray(reference, float).copy()
        missed = np.ones(MARKER_COUNT, int)
        for index, point in enumerate(found):
            if point is not None:
                points[index] = point
                missed[index] = 0
        self.last_points = points
        self.missed = missed
        self.overlapped[:] = False
        self.hue_offset = offset
        count = int(np.count_nonzero(missed == 0))
        return points.copy(), f"{count}/10 reacquired; {MARKER_COUNT-count} held"

    def _plausible_partial(self, found):
        known = [(index, point) for index, point in enumerate(found) if point is not None]
        if len(known) < 7:
            return False
        if any(right[1][0]-left[1][0] <= 4 for left, right in zip(known, known[1:])):
            return False
        return known[-1][1][0] < self.settings["base"][0]

    def _distinct_blue_components(self, frame, points, offset):
        """Require separate image blobs before ending an overlap hold."""
        target = ((self.settings["blue_hsv"][0]+offset) % 180,
                  *self.settings["blue_hsv"][1:])
        mask = color_mask(frame, target, 13, self.settings["roi"])
        _, labels = cv2.connectedComponents(mask)
        ids = []
        height, width = labels.shape
        for x, y in points[1:]:
            x, y = int(round(x)), int(round(y))
            patch = labels[max(0, y-4):min(height, y+5),
                           max(0, x-4):min(width, x+5)]
            nonzero = patch[patch != 0]
            if not len(nonzero):
                return False
            ids.append(int(np.argmax(np.bincount(nonzero))))
        return len(set(ids)) == len(ids)

    def _recover(self, frame, allow_partial=True):
        """Reacquire after a bend, including a shifted return pose."""
        references = []
        best_partial = None
        best_count = 0
        def consider(found, reference, offset):
            nonlocal best_partial, best_count
            count = sum(point is not None for point in found)
            if count > best_count and self._plausible_partial(found):
                best_count = count
                best_partial = (found, reference, offset)
        for reference in (self.last_good_points, self.settings.get("reference_points")):
            if reference is not None:
                reference = np.asarray(reference, float)
                if not any(np.allclose(reference, other) for other in references):
                    references.append(reference)
        roi = self.settings["roi"]
        for offset in dict.fromkeys((self.hue_offset, 0, 10, -10, 20, -20)):
            blue_target = ((self.settings["blue_hsv"][0]+offset) % 180,
                           *self.settings["blue_hsv"][1:])
            green_target = ((self.settings["green_hsv"][0]+offset) % 180,
                            *self.settings["green_hsv"][1:])
            masks = (color_mask(frame, blue_target, 13, roi),
                     color_mask(frame, green_target, 9, roi))
            contours, _ = cv2.findContours(masks[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tips = []
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                if not (100 <= area <= 3000 and w >= 8 and h >= 8
                        and max(w/h, h/w) <= 2.5):
                    continue
                moments = cv2.moments(contour)
                if moments["m00"]:
                    tips.append(np.array([moments["m10"]/moments["m00"],
                                          moments["m01"]/moments["m00"]]))
            for reference in references:
                for tip in sorted(tips, key=lambda point: np.linalg.norm(point-reference[0])):
                    shift = tip-reference[0]
                    if np.linalg.norm(shift) > 180:
                        continue
                    translated = reference+shift
                    found = follow_markers(frame, self.settings, translated, offset, masks)
                    if all(point is not None for point in found) and self._plausible_chain(found, reference):
                        return self._accept_recovery(found, offset)
                    consider(found, translated, offset)
                found = follow_markers(frame, self.settings, reference, offset, masks)
                if all(point is not None for point in found) and self._plausible_chain(found, reference):
                    return self._accept_recovery(found, offset)
                consider(found, reference, offset)
        best, status = [], "markers 0/10"
        for offset in dict.fromkeys((self.hue_offset, 0, 5, -5, 10, -10, 15, -15, 20, -20)):
            candidate, candidate_status = marker_chain(frame, self.settings, None, offset)
            if len(candidate) > len(best):
                best, status = candidate, candidate_status
            plausible = (any(self._plausible_chain(candidate, reference)
                             for reference in references) if references else
                         self._plausible_chain(candidate))
            if plausible and (allow_partial or self._distinct_blue_components(frame, candidate, offset)):
                return self._accept_recovery(candidate, offset)
        if allow_partial and best_partial is not None:
            return self._accept_partial_recovery(*best_partial)
        return None, status

    def _current_hue_offset(self, frame):
        """Sample known blue discs so camera auto exposure cannot stale the saved hue."""
        if self.last_points is None:
            return self.hue_offset
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        hues = []
        for index, (x, y) in enumerate(self.last_points[1:], 1):
            if self.missed is not None and self.missed[index]:
                continue
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
        detected = list(detected)
        overlap_pairs = np.zeros(MARKER_COUNT-1, bool)
        for index in range(1, MARKER_COUNT-1):
            left, right = detected[index:index+2]
            if left is not None and right is not None and np.linalg.norm(np.subtract(left, right)) < 4:
                overlap_pairs[index] = True
        if all(point is not None for point in detected) and not self._plausible_chain(
                detected, self.last_good_points, overlap_pairs):
            reference = self.last_good_points if self.last_good_points is not None else self.last_points
            candidate = np.asarray(detected, float)
            shift = np.median(candidate-reference, axis=0)
            errors = np.linalg.norm(candidate-reference-shift, axis=1)
            gaps = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
            reference_gaps = np.linalg.norm(np.diff(reference, axis=0), axis=1)
            conflicts = np.flatnonzero(gaps < .45*reference_gaps)
            if len(conflicts):
                pair = int(conflicts[np.argmin(gaps[conflicts])])
                rejected = pair + int(errors[pair+1] >= errors[pair])
            else:
                rejected = int(np.argmax(errors))
            detected[rejected] = None
        updated = self.last_points.copy()
        visible = 0
        for index, point in enumerate(detected):
            if point is not None and np.linalg.norm(self.last_points[index]-point) <= 45:
                updated[index] = (np.asarray(point) if (index > 0 and overlap_pairs[index-1]) or
                                  (index < MARKER_COUNT-1 and overlap_pairs[index]) else
                                  .65*self.last_points[index]+.35*np.asarray(point))
                self.missed[index] = 0
                visible += 1
            else:
                self.missed[index] += 1
        if visible == MARKER_COUNT and not self._plausible_chain(
                updated, self.last_good_points, overlap_pairs):
            reference = self.last_good_points if self.last_good_points is not None else self.last_points
            rejected = int(np.argmax(np.linalg.norm(updated-reference, axis=1)))
            updated[rejected] = self.last_points[rejected]
            self.missed[rejected] += 1
            visible -= 1
        self.last_points = updated
        self.overlapped[:] = False
        for index, overlapping in enumerate(overlap_pairs):
            if overlapping and self.missed[index] == self.missed[index+1] == 0:
                self.overlapped[index:index+2] = True
        if visible == MARKER_COUNT and not np.any(self.overlapped):
            self.last_good_points = updated.copy()
            self.recovery_ticks = 0
        return updated.copy(), (f"10/10; {int(np.count_nonzero(self.overlapped))} overlapped"
                                if np.any(self.overlapped) else "10/10" if visible == 10 else
                                f"{visible}/10 visible; {10-visible} held")

    def draw(self, image, frame, _depth=None, _vision=None):
        if not self.enabled:
            return image
        if self.shape != image.shape[:2]:
            self.shape = image.shape[:2]
            self.settings = load_settings(image.shape, self.config_path)
            self.last_points = self.last_good_points = self.missed = None
            self.overlapped[:] = False
            self.hue_offset = 0
            self.recovery_ticks = 0
            if self.settings is not None and self.settings.get("reference_points") is not None:
                self.last_points = np.asarray(self.settings["reference_points"], float)
                self.last_good_points = self.last_points.copy()
                self.missed = np.zeros(MARKER_COUNT, int)
        if self.settings is None:
            self._text(image, "Bending: run `python nus_main.py bending-setup`", 0)
            return image
        offset = self._current_hue_offset(frame)
        if self.last_points is not None:
            detected = follow_markers(frame, self.settings, self.last_points, offset)
            detection_status = f"local {sum(p is not None for p in detected)}/10"
            points, status = self._stable_indexed(detected)
            if points is None:
                points, status = self._recover(frame)
            elif (self.missed is not None and np.any(self.missed >= 5)) or np.any(self.overlapped):
                self.recovery_ticks += 1
                if self.recovery_ticks >= 3:
                    self.recovery_ticks = 0
                    recovered, recovery_status = self._recover(
                        frame, allow_partial=not np.any(self.overlapped))
                    if recovered is not None:
                        points, status = recovered, recovery_status
        else:
            points, status = self._recover(frame)
            detection_status = status
        if points is None:
            reference = self.settings.get("reference_points", [])
            draw_reference_markers(image, reference)
            self._text(image, "Bending: " + status + " (" + detection_status + ")", 0)
            self._text(image, "Reference markers shown; press m in RGB preview to edit", 1)
            return image
        base = tuple(self.settings["base"])
        if base[0] <= points[-1][0]:
            self._text(image, "Bending: base must be right of markers", 0)
            return image
        estimated_index = None
        estimated = None
        if self.missed is not None and np.any(self.missed) and not np.any(self.overlapped):
            reference = (self.last_good_points if self.last_good_points is not None else
                         self.settings.get("reference_points"))
            if reference is not None:
                estimated = estimate_hidden_blue(points, self.missed, base, reference)
        if estimated is not None:
            points, estimated_index = estimated
        ordered = [base] + [tuple(point) for point in points[::-1]]
        angle_unavailable = (np.any(self.overlapped) or
                             (self.missed is not None and np.any(self.missed) and
                              estimated_index is None))
        if angle_unavailable:
            sections_2d = []
            two_d = "angle paused until markers separate" if np.any(self.overlapped) else \
                    "angle paused until all markers are visible"
        else:
            try:
                turning_angles(ordered)
            except ValueError:
                if estimated_index is not None:
                    sections_2d = []
                    two_d = "2D estimate unavailable for this pose"
                else:
                    self.last_points = self.missed = None
                    self._text(image, "Bending: marker assignments conflicted; reacquiring", 0)
                    return image
            else:
                sections_2d = sorted(
                    sorted(bend_sections(ordered, min_bend=5.0),
                           key=lambda section: abs(section[2]), reverse=True)[:2],
                    key=lambda section: section[0],
                )
                prefix = "~" if estimated_index is not None else ""
                two_d = (("2D estimated " if prefix else "2D ") +
                         (" | ".join(f"B{i} {prefix}{angle:+.1f} deg"
                                   for i, (_, _, angle) in enumerate(sections_2d, 1))
                          if sections_2d else "straight"))
        visible_chain = [True] + ([self.missed[index] == 0 for index in range(MARKER_COUNT-1, -1, -1)]
                                  if self.missed is not None else [True]*MARKER_COUNT)
        if estimated_index is not None:
            visible_chain[MARKER_COUNT-estimated_index] = True
        for index in range(len(ordered)-1):
            if visible_chain[index] and visible_chain[index+1]:
                start = tuple(np.rint(ordered[index]).astype(int))
                end = tuple(np.rint(ordered[index+1]).astype(int))
                cv2.line(image, start, end, (0, 0, 0), 6)
                color = ((0, 165, 255) if estimated_index is not None and
                         MARKER_COUNT-estimated_index in (index, index+1) else
                         (0, 255, 255))
                cv2.line(image, start, end, color, 3)
        cv2.circle(image, tuple(map(int, base)), 12, (0, 0, 255), 3)
        for index, point in enumerate(ordered[1:], 1):
            center = tuple(map(int, point))
            marker_index = MARKER_COUNT-index
            color = ((0, 165, 255) if marker_index == estimated_index else
                     (140, 140, 140) if not visible_chain[index] else
                     (255, 255, 0) if self.overlapped[marker_index] else
                     (0, 255, 0) if index == MARKER_COUNT else (255, 100, 0))
            cv2.circle(image, center, 13, (0, 0, 0), 6)
            cv2.circle(image, center, 13, color, 4)
            cv2.putText(image, "T" if index == MARKER_COUNT else
                        f"~B{marker_index}" if marker_index == estimated_index else
                        f"B{marker_index}",
                        (center[0]+11, center[1]-10+(16 if self.overlapped[marker_index]
                                                    and marker_index % 2 else 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45,
                        color, 2)
        for start, end, angle in sections_2d:
            if abs(angle) <= 5.0:
                continue
            center_index = (start + end) // 2
            x, y = np.rint(ordered[center_index]).astype(int)
            label = f"{'~' if estimated_index is not None else ''}{angle:+.1f} deg"
            x = int(np.clip(x + 14, 4, max(4, image.shape[1]-125)))
            y = int(np.clip(y - 22, 25, max(25, image.shape[0]-65)))
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .68, (0, 0, 0), 5)
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .68, (0, 165, 255) if estimated_index is not None else
                        (0, 255, 255), 2)
        self._text(image, f"Bending: {status} | {two_d}", 0)
        return image

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


def add_manual_marker(points, kind, point, roi):
    """Build a tip-to-base reference chain from explicit user clicks."""
    x, y = point
    if roi is None or not (roi[0] <= x < roi[0]+roi[2] and roi[1] <= y < roi[1]+roi[3]):
        raise ValueError("Click inside the ROI")
    if kind == "g":
        return [(float(x), float(y))]
    if not points:
        raise ValueError("Click green tip first (g)")
    if len(points) >= MARKER_COUNT:
        raise ValueError("All 9 blue markers are set; u undo or g restart")
    if x <= points[-1][0]:
        raise ValueError("Click blue markers from tip toward base (left to right)")
    return points + [(float(x), float(y))]


def draw_reference_markers(image, points, prefix=""):
    """Show the exact saved/clicked centers even if color tracking fails."""
    for index, point in enumerate(points):
        center = tuple(map(lambda v: int(round(v)), point))
        color = (0, 255, 0) if index == 0 else (255, 100, 0)
        label = "T" if index == 0 else f"B{index}"
        cv2.circle(image, center, 13, (0, 0, 0), 5)
        cv2.circle(image, center, 13, color, 2)
        cv2.putText(image, prefix + label, (center[0]+12, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
        cv2.putText(image, prefix + label, (center[0]+12, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)


def run_setup(camera, config_path=CONFIG_PATH):
    """Interactive ROI, base, and one-by-one marker calibration."""
    window = "Bending setup (r ROI, a base, g tip, b blue x9, u undo, s save, q quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    existing = None
    roi = base = None
    blue_hsv, green_hsv = BLUE_HSV, GREEN_HSV
    manual_points = []
    mode = None
    frame = None
    note = "r ROI; a base; g green tip; b blue markers from tip to base"

    def persist():
        nonlocal note
        if roi is None or base is None:
            note = "Select ROI (r) and base (a) before saving"
            return
        try:
            save_settings(frame.shape, roi, base, blue_hsv, green_hsv, config_path,
                          reference_points=manual_points if len(manual_points) == MARKER_COUNT else None)
            note = ("Saved T + B1-B9 reference points" if len(manual_points) == MARKER_COUNT
                    else f"Saved ROI/base/colors; click {MARKER_COUNT-len(manual_points)} more markers")
        except OSError as exc:
            note = f"Could not save calibration: {exc}"

    def mouse(event, x, y, _flags, _param):
        nonlocal mode, base, blue_hsv, green_hsv, manual_points, note
        if event != cv2.EVENT_LBUTTONDOWN or frame is None or mode is None:
            return
        if mode == "a":
            base = (x, y)
            persist()
        elif mode in ("b", "g"):
            try:
                updated = add_manual_marker(manual_points, mode, (x, y), roi)
                try:
                    value = sample_color(frame, (x, y))
                except ValueError:
                    value = None  # Keep the clicked position even in washed-out light.
                manual_points = updated
                if mode == "g" and value is not None: green_hsv = value
                elif len(manual_points) == 2 and value is not None: blue_hsv = value
                persist()
                if mode == "g":
                    note = "T set; press b and click blue markers from tip to base"
                elif len(manual_points) < MARKER_COUNT:
                    note = f"B{len(manual_points)-1} set; click B{len(manual_points)}"
            except ValueError as exc:
                note = str(exc)
                return
        if mode != "b" or len(manual_points) == MARKER_COUNT:
            mode = None

    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            camera.update()
            frame = camera.frame.copy()
            if existing is None:
                existing = load_settings(frame.shape, config_path)
                if existing:
                    roi, base = tuple(existing["roi"]), tuple(existing["base"])
                    blue_hsv, green_hsv = tuple(existing["blue_hsv"]), tuple(existing["green_hsv"])
                    manual_points = [tuple(point) for point in existing.get("reference_points", [])]
                    note = "Saved calibration loaded; g restarts markers, b adds blue, u undoes"
            view = frame.copy()
            if roi: cv2.rectangle(view, roi[:2], (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 255), 2)
            if base: cv2.drawMarker(view, base, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            draw_reference_markers(view, manual_points)
            BendingOverlay._text(view, note, 0)
            BendingOverlay._text(view, f"Manual markers {len(manual_points)}/10 | r ROI  a base  g tip  b blue  u undo  s save  q quit", 1)
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), 27): break
            if key in (ord("a"), ord("b"), ord("g")):
                mode = chr(key)
                note = ("Click base" if mode == "a" else
                        "Click green tip to start a new chain" if mode == "g" else
                        f"Click B{len(manual_points)} from tip toward base")
            if key == ord("u"):
                if manual_points:
                    manual_points.pop()
                    note = f"Undid marker; {len(manual_points)}/10 remain"
                    persist()
                mode = None
            if key == ord("s"):
                persist()
            if key == ord("r"):
                selected = cv2.selectROI(window, frame, False, False)
                if selected[2] and selected[3]:
                    roi = tuple(int(v) for v in selected)
                    manual_points = []
                    note = "ROI changed; click g tip, then b for nine blue markers"
                    persist()
                cv2.setMouseCallback(window, mouse)
    finally:
        cv2.destroyWindow(window)


class RGBCamera:
    """OpenCV color camera for 2D setup and preview."""
    def __init__(self, index=0):
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open RGB camera {index}")
        try:
            width, height = json.loads(CONFIG_PATH.read_text())["frame_size"]
        except (OSError, ValueError, TypeError, KeyError):
            width, height = 1280, 720
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None

    def update(self):
        ok, self.frame = self.capture.read()
        if not ok or self.frame is None:
            raise RuntimeError("Could not read RGB frame")

    def close(self):
        self.capture.release()


def run_preview(camera, config_path=CONFIG_PATH):
    """Preview marker tracking without motor control."""
    window = "Bending RGB preview (m setup, v toggle, q quit)"
    print(f"Opening preview window: {window}", flush=True)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    overlay = BendingOverlay(enabled=True, config_path=config_path)
    try:
        while True:
            camera.update()
            frame = camera.frame.copy()
            overlay.draw(frame, camera.frame, None, None)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), 27):
                break
            if key == ord("v"):
                overlay.enabled = not overlay.enabled
            if key == ord("m"):
                run_setup(camera, config_path)
                overlay.shape = None  # Reload the saved clicks on the next frame.
    finally:
        cv2.destroyWindow(window)
