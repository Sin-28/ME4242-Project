"""Optional, read-only bending overlay for the existing EMIO color/depth frames."""

import json
import math
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
        return settings
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return None


def save_settings(shape, roi, base, blue_hsv, green_hsv, path=CONFIG_PATH):
    height, width = shape[:2]
    data = {"frame_size": [width, height], "roi": list(roi), "base": list(base),
            "blue_hsv": list(blue_hsv), "green_hsv": list(green_hsv)}
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def color_mask(frame, target, tolerance, roi):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = target
    distance = abs(hsv[..., 0].astype(np.int16) - hue)
    distance = np.minimum(distance, 180 - distance)
    mask = ((distance <= tolerance) & (hsv[..., 1] >= max(35, int(sat * .45)))
            & (hsv[..., 2] >= max(20, int(val * .3)))).astype(np.uint8) * 255
    x, y, w, h = roi
    outside = np.zeros_like(mask)
    outside[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
    return cv2.morphologyEx(outside, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def marker_chain(frame, settings):
    """Locate colored discs along image x; returns left-to-right centers."""
    roi = settings["roi"]
    x0, y0, width, height = roi
    mask = cv2.bitwise_or(color_mask(frame, settings["blue_hsv"], 15, roi),
                          color_mask(frame, settings["green_hsv"], 7, roi))
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
    if len(peaks) > MARKER_COUNT:
        from itertools import combinations
        strengths = {x: float(projection[x]) for x in peaks}
        def regularity(selected):
            gaps = np.diff(sorted(selected)).astype(float)
            return float(np.sum(np.diff(np.log(gaps))**2)) - .001 * sum(strengths[x] for x in selected)
        peaks = list(min(combinations(peaks, MARKER_COUNT), key=regularity))
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
    if len(points) < 7:
        return [], f"markers {len(points)}/{MARKER_COUNT}"
    return points, f"markers {len(points)}/{MARKER_COUNT}"


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

    def draw(self, image, frame, depth, vision):
        if not self.enabled:
            return image
        if self.shape != image.shape[:2]:
            self.shape = image.shape[:2]
            self.settings = load_settings(image.shape)
        if self.settings is None:
            self._text(image, "Bending: run `python nus_main.py bending-setup`", 0)
            return image
        points, status = marker_chain(frame, self.settings)
        if not points:
            self._text(image, "Bending: " + status, 0)
            return image
        base = tuple(self.settings["base"])
        if base[0] <= points[-1][0]:
            self._text(image, "Bending: base must be right of markers", 0)
            return image
        ordered = [base] + points[::-1]
        try:
            turns = turning_angles(ordered)
        except ValueError as exc:
            self._text(image, "Bending: " + str(exc), 0)
            return image
        bend = split_two_bends(turns)
        two_d = (f"2D base {bend[1]:+.1f} | tip {bend[2]:+.1f} deg" if bend
                 else f"2D bend {turns.sum():+.1f} deg")
        three_d, depth_status = spatial_angles(ordered, bend, depth, vision.pixel_to_robot)
        cv2.polylines(image, [np.asarray(ordered, np.int32)], False, (255, 255, 255), 2)
        for point in ordered[1:]:
            cv2.circle(image, tuple(map(int, point)), 4, (0, 255, 255), -1)
        self._text(image, f"Bending: {status} | {two_d}", 0)
        self._text(image, depth_status, 1)
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
                save_settings(frame.shape, roi, base, blue_hsv, green_hsv)
                note = "Saved calibration; q to quit or adjust selections"
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
            if key == ord("r"):
                selected = cv2.selectROI(window, frame, False, False)
                if selected[2] and selected[3]:
                    roi = tuple(int(v) for v in selected)
                    persist()
                cv2.setMouseCallback(window, mouse)
    finally:
        cv2.destroyWindow(window)
