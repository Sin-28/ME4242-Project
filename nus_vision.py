"""
Vision for the N-U-S retrieval task: green-tip tracking, N/U/S letter detection,
and pixel -> robot coordinates.

Robot frame (from emioapi's camera calibration): millimetres, y is vertical (up),
x and z are horizontal. The bare EMIO platform is at y = -303 mm.

Settings live in nus_config.json (created with defaults on first run).
Taught letter shapes live in letter_templates.npz (created by `nus_main.py detect`).

Two letter modes (`letter_mode` in nus_config.json):
  - "color" (default): each letter has its own colour (N red, U black, S blue). Each colour is
    segmented separately, and every blob is shape-checked against that letter's template so
    same-coloured things (the blue arm disks, black cables) are rejected. Learn the colours on
    site: `nus_main.py detect`, click a letter, press n / u / s.
  - "shape": letters of one colour, found as blobs that differ from `background_hsv` and
    told apart by shape (e.g. white letters on black cloth: background [[[0, 0, 0], [180, 255, 100]]]).
`background_hsv` is also the surface `check` measures the height of (default: white platform).
Only pixels within `max_radius_mm` of the robot axis (on the stage plane) are searched,
so things beside the robot (e.g. a computer case) are ignored.
"""

import copy
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from nus_path import convex_hull, segment_distance

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "nus_config.json"
TEMPLATE_PATH = HERE / "letter_templates.npz"               # shape mode (letters taught on the black cloth)
COLOR_TEMPLATE_PATH = HERE / "letter_templates_color.npz"   # colour mode (cleaner outlines, kept separate)
LETTERS = ("N", "U", "S")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# HSV ranges are lists of [[h, s, v] low, [h, s, v] high] pairs (OpenCV: H 0-180, S/V 0-255).
DEFAULT_CONFIG = {
    "letter_mode": "color",          # "color": one colour per letter; "shape": one colour, told apart by shape
    "letter_hsv": {                  # colour mode. Learn on site: `detect`, click a letter, press n / u / s
        "N": [[[0, 100, 60], [10, 255, 255]], [[165, 100, 60], [180, 255, 255]]],   # red (hue wraps at 180)
        "U": [[[0, 0, 0], [180, 255, 70], 30]],                                     # black: dark, chroma <= 30
        "S": [[[95, 120, 40], [130, 255, 255]]],                                    # blue
    },
    "color_min_overlap": 0.45,       # colour mode: shape check against that letter's template (0 = off)
    "color_max_ellipse_fit": 0.8,    # colour mode: blobs this close to an ellipse are round things (arm disks)
    "color_min_aspect": 0.4,         # colour mode: thinner blobs (short side / long side) are cables or edges
    "color_max_height_mm": 12.0,     # colour mode: a blob whose top is this far above the surface right around
                                     #   it is on the arm (letters lie flat, at any height: platform or a block)
    "color_max_spread_mm": 7.0,      # colour mode: a letter is flat; a blob spanning more heights is an object's side
    "letter_thickness_mm": 5.0,
    "tip_length_mm": 8.0,            # the magnet tip sticks out this far past the green marker the camera tracks
    "holder_slots": {},              # robot (x, y, z) of each letter's slot in the holder: `python nus_main.py holder`
    "obstacle_height_mm": 8.0,       # anything (any colour) standing this high above the work surface is an
                                     #   obstacle: depth camera, above the letters' thickness + depth noise
    "obstacle_max_height_mm": 150.0, # ...unless part of it reaches higher: then it's a hand or the robot
    "arm_radius_mm": 20.0,           # points this close to the line robot axis -> tip (above the tip) are the arm
    "obstacle_min_area_px": 500,     # smaller blobs are depth noise on the white platform (~11 x 11 mm at 20 cm)
    "obstacle_grey_max_saturation": 80,   # pixels this unsaturated (white/grey/black) are grouped apart from coloured ones
    "magnet_gap_px": 18,             # a magnet can cut a letter's stroke: pieces this close are joined back up
    "magnet_area_px": [15, 600],     # a hole of this size inside a letter is its magnet (= the point to go to)...
    "magnet_min_circularity": 0.6,   # ...if it is round (1 = circle); an S's closed-off inner curve is not
    "shape_min_overlap": 0.6,        # shape mode: best overlap (IoU) with a template to count as a letter
    "min_margin": 0.05,              # shape mode: best letter must beat the runner-up by this much, else "?"
    "tip_hsv": [[[40, 100, 60], [84, 255, 255]]],      # green tip (measured hue 73-81)
    "robot_hsv": [                   # shape mode: never a letter (green tip, blue disks, teal legs)
        [[40, 100, 40], [84, 255, 255]],
        [[86, 80, 30], [125, 255, 255]],
    ],
    "background_hsv": [[[0, 0, 150], [180, 60, 255]]],   # surface under the letters: white platform
    "stage_y_mm": -228.0,            # main surface + letter thickness: only for the search circle and when a
                                     #   blob has no depth readings (heights otherwise come from the depth camera)
    "max_radius_mm": 130.0,          # only search this far from the robot axis (on the stage plane)
    "min_area_px": 150,
    "max_area_px": 20000,
    "tip_min_area_px": 30,
    "tip_max_jump_px": 120,          # tip can't jump further than this between frames
    "tip_max_radius_mm": 170.0,      # green blobs further from the robot axis are background objects
    "tip_y_range_mm": [-330.0, -80.0],   # ...and so are green blobs outside the arm's height range
}


def load_config():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    else:
        _write_config(DEFAULT_CONFIG)
    return cfg


def save_config_value(key, value):
    """Change one setting in nus_config.json, keeping the others as they are."""
    values = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else copy.deepcopy(DEFAULT_CONFIG)
    values[key] = value
    _write_config(values)


def _write_config(values):
    # one setting per line, so the HSV ranges are easy to edit
    CONFIG_PATH.write_text("{\n" + ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}"
                                             for k, v in values.items()) + "\n}\n")


def fmt(p):
    return "none" if p is None else f"({p[0]:+.0f}, {p[1]:+.0f}, {p[2]:+.0f}) mm"


def hsv_mask(hsv, ranges):
    """Union of several HSV ranges. A range may have a third value, a maximum chroma
    (S * V / 255): needed for black, because saturation is pure noise on very dark pixels
    while chroma (absolute colourfulness) stays small."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    chroma = None
    for entry in ranges:
        low, high = entry[0], entry[1]
        m = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
        if len(entry) > 2:
            if chroma is None:
                chroma = (hsv[..., 1].astype(np.uint16) * hsv[..., 2]) // 255
            m &= ((chroma <= entry[2]) * 255).astype(np.uint8)
        mask |= m
    return mask


def learn_hsv_range(hsv_patch, hue_margin=8, sat_margin=40, val_margin=50):
    """HSV range(s) covering a patch of one colour, plus margins. Handles red's hue wrap-around
    (returns two ranges). Dark or unsaturated colours (black, grey) get any hue and saturation,
    limited by brightness and chroma instead (saturation is noise on dark pixels)."""
    h, s, v = (hsv_patch[..., i].ravel().astype(float) for i in range(3))
    low_sv = [int(max(0, np.percentile(s, 5) - sat_margin)), int(max(0, np.percentile(v, 5) - val_margin))]
    high_sv = [int(min(255, np.percentile(s, 95) + sat_margin)), int(min(255, np.percentile(v, 95) + val_margin))]
    if np.median(s * v / 255) < 35:                         # black / grey / white: hue is meaningless
        chroma_max = int(min(255, max(20, np.percentile(s * v / 255, 95) + 15)))
        return [[[0, 0, low_sv[1]], [180, 255, high_sv[1]], chroma_max]]
    angle = h * np.pi / 90                                   # hue 0-180 -> angle 0-2pi (hue is circular)
    centre = (np.degrees(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())) / 2) % 180
    half = np.percentile(np.abs((h - centre + 90) % 180 - 90), 95) + hue_margin
    low, high = int(np.floor(centre - half)), int(np.ceil(centre + half))
    if low < 0:
        return [[[0, *low_sv], [high, *high_sv]], [[180 + low, *low_sv], [180, *high_sv]]]
    if high > 180:
        return [[[low, *low_sv], [180, *high_sv]], [[0, *low_sv], [high - 180, *high_sv]]]
    return [[[low, *low_sv], [high, *high_sv]]]


def contour_centre(contour):
    m = cv2.moments(contour)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


def _largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None


def default_templates():
    """Bold sans-serif N, U, S outlines (two OpenCV fonts). Serif or unusual fonts need the real
    letters taught: generic templates for many fonts made the letters look alike and hurt accuracy."""
    templates = {}
    for letter in LETTERS:
        templates[letter] = []
        for font, thickness in ((cv2.FONT_HERSHEY_SIMPLEX, 34), (cv2.FONT_HERSHEY_DUPLEX, 30)):
            canvas = np.zeros((300, 300), np.uint8)
            cv2.putText(canvas, letter, (50, 240), font, 7, 255, thickness)
            templates[letter].append(_largest_contour(canvas))
    return templates


NORM_SIZE = 96                         # shapes are compared on a 96 x 96 canvas
NORM_FILL = 0.25                       # ...scaled so the shape covers 25% of it
ROTATIONS = range(0, 360, 10)


def normalised_masks(contour, angles=(0,), mirror=False):
    """Filled shape, centred on its centroid and scaled to a fixed area, at each rotation angle."""
    x, y, w, h = cv2.boundingRect(contour)
    local = np.zeros((h, w), np.uint8)
    cv2.drawContours(local, [contour - [x, y]], -1, 255, -1)
    if mirror:
        local = cv2.flip(local, 1)
    m = cv2.moments(local, binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    scale = np.sqrt(NORM_FILL * NORM_SIZE ** 2 / m["m00"])
    masks = []
    for angle in angles:
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        M[:, 2] += (NORM_SIZE / 2 - cx, NORM_SIZE / 2 - cy)
        masks.append(cv2.warpAffine(local, M, (NORM_SIZE, NORM_SIZE), flags=cv2.INTER_NEAREST) > 0)
    return masks


def best_overlap(blob_masks, template_masks):
    """Highest intersection-over-union between any blob rotation and any template."""
    best = 0.0
    for b in blob_masks:
        for t in template_masks:
            union = np.count_nonzero(b | t)
            if union:
                best = max(best, np.count_nonzero(b & t) / union)
    return best


def ellipse_fit(contour):
    """Overlap (IoU) between a blob and its best-fit ellipse: ~0.9 for a disk, ~0.5-0.75 for a letter."""
    if len(contour) < 5:
        return 1.0
    x, y, w, h = cv2.boundingRect(contour)
    pad = 4
    local = contour - [x - pad, y - pad]
    blob = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.drawContours(blob, [local], -1, 255, -1)
    ellipse = np.zeros_like(blob)
    cv2.ellipse(ellipse, cv2.fitEllipse(local), 255, -1)
    return np.count_nonzero(blob & ellipse) / max(np.count_nonzero(blob | ellipse), 1)


def enclosed_pixels(mask):
    """Pixels not in the mask but fully surrounded by it (the mask's holes)."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)   # so (0, 0) is outside
    reached = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)       # floodFill's mask
    cv2.floodFill(padded.copy(), reached, (0, 0), 255, flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    holes = (padded == 0) & (reached[1:-1, 1:-1] == 0)     # background the outside fill couldn't reach
    return holes[1:-1, 1:-1]


def aspect_ratio(contour):
    """Short side / long side of the rotated bounding box: low for cables and edges."""
    short, long = sorted(cv2.minAreaRect(contour)[1])
    return short / max(long, 1e-6)


def _load_taught(path=TEMPLATE_PATH):
    taught = {}
    if path.exists():
        with np.load(path) as data:
            for key in sorted(data.files):
                taught.setdefault(key.split("_")[0], []).append(data[key])
    return taught


class Vision:
    def __init__(self, cfg, camera=None):
        """camera: an opened emioapi EmioCamera, or None to run the detectors on saved images."""
        self.cfg = cfg
        self.camera = camera
        self.color_mode = cfg["letter_mode"] == "color"
        self.template_path = COLOR_TEMPLATE_PATH if self.color_mode else TEMPLATE_PATH
        self.taught = _load_taught(self.template_path)
        defaults = default_templates()
        if self.color_mode:
            # colour already says which letter it is, so extra templates only make acceptance more
            # tolerant: keep the font templates alongside the taught ones
            self.templates = {L: defaults[L] + self.taught.get(L, []) for L in LETTERS}
        else:
            # shape mode tells letters apart by shape: a taught letter replaces its font template
            self.templates = {**defaults, **self.taught}
        self._template_masks = {L: self._masks_for(self.templates[L]) for L in LETTERS}
        self._roi = None
        self._rays = None
        self.last_tip_px = None
        self._tip_misses = 0

    # ---------------- camera ----------------

    @staticmethod
    def open_camera():
        from emioapi import EmioCamera
        # track_markers=False: the built-in tracker draws text onto the frame
        camera = EmioCamera(show=False, track_markers=False, compute_point_cloud=False)
        return camera if camera.open() else None

    def grab(self):
        self.camera.update()
        return self.camera.frame.copy(), self.camera.depth_frame.copy()

    # ---------------- pixel -> robot frame ----------------

    def pixel_to_robot(self, u, v, depth_mm):
        return np.array(self.camera.image_to_simulation(int(u), int(v), float(depth_mm)), float)

    def pixel_to_plane(self, u, v, plane_y):
        """Where the camera ray through pixel (u, v) meets the horizontal plane y = plane_y.
        Used when the depth camera has no reading (common on flat, untextured surfaces)."""
        if self.camera is None:
            return None
        est = self.camera._camera.position_estimator   # emioapi internals: camera -> robot transform
        intr = est.intr
        ray = np.array([(u - intr.ppx) / intr.fx, (v - intr.ppy) / intr.fy, 1.0])
        denom = est.R[1] @ ray
        if abs(denom) < 1e-9:
            return None
        d = (plane_y - est.t[1]) / denom
        return est.R @ (ray * d) + est.t if d > 0 else None

    def pixels_to_robot(self, us, vs, ds):
        """image_to_simulation for many pixels at once (arrays of u, v, depth mm) -> N x 3."""
        est = self.camera._camera.position_estimator
        intr = est.intr
        cam = np.stack([(us - intr.ppx) / intr.fx * ds, (vs - intr.ppy) / intr.fy * ds, ds], axis=-1)
        return cam @ est.R.T + est.t

    def robot_to_pixel(self, point):
        """Where a robot-frame point appears in the image (for drawing), or None if behind the camera."""
        if self.camera is None:
            return None
        est = self.camera._camera.position_estimator
        intr = est.intr
        cam = est.R.T @ (np.asarray(point, float) - est.t)        # R is a rotation: inverse = transpose
        if cam[2] <= 1e-6:
            return None
        return np.array([intr.fx * cam[0] / cam[2] + intr.ppx, intr.fy * cam[1] / cam[2] + intr.ppy])

    def support_height(self, blob_mask, depth, ring_px=8):
        """Height (robot y) of the surface right around a blob, from the depth of a ring of pixels just
        outside it. This is what lets letters lie at any height (platform, block, ...): a letter is a
        few mm above it, an arm disk far above. None if there are too few depth readings."""
        if self.camera is None:
            return None
        outer = cv2.dilate(blob_mask, np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8))
        inner = cv2.dilate(blob_mask, np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero((outer > 0) & (inner == 0) & (depth > 0))
        if len(xs) < 20:
            return None
        points = self.pixels_to_robot(xs.astype(float), ys.astype(float), depth[ys, xs].astype(float))
        return float(np.median(points[:, 1]))

    def blob_height(self, contour, depth):
        """(top of the blob, surface around it) as robot y in mm; either can be None."""
        if self.camera is None:
            return None, None
        mask = np.zeros(depth.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        d = depth[(mask > 0) & (depth > 0)]
        if len(d) < 0.3 * max(np.count_nonzero(mask), 1):
            top = None
        else:
            top = float(self.pixel_to_robot(*contour_centre(contour), np.median(d))[1])
        return top, self.support_height(mask, depth)

    def surface_height(self, frame, depth, radius_mm=100.0, max_points=3000):
        """Median height (robot y, mm) of background-coloured pixels within radius_mm of the
        robot axis, measured with the depth camera. Returns (height or None, points used)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ys, xs = np.nonzero((hsv_mask(hsv, self.cfg["background_hsv"]) > 0) & (depth > 0))
        if len(xs) < 50:
            return None, len(xs)
        pick = np.random.default_rng(0).choice(len(xs), min(max_points, len(xs)), replace=False)
        xs, ys = xs[pick].astype(float), ys[pick]
        est = self.camera._camera.position_estimator
        intr = est.intr
        d = depth[ys, xs.astype(int)].astype(float)
        cam = np.stack([(xs - intr.ppx) / intr.fx * d, (ys - intr.ppy) / intr.fy * d, d], axis=-1)
        points = cam @ est.R.T + est.t                   # same transform as image_to_simulation
        near = points[np.hypot(points[:, 0], points[:, 2]) < radius_mm]
        if len(near) < 50:
            return None, len(near)
        return float(np.median(near[:, 1])), len(near)

    def points_map(self, depth):
        """Robot-frame point (x, y, z) of every pixel, H x W x 3 (NaN where there is no depth reading)."""
        est = self.camera._camera.position_estimator
        intr = est.intr
        if self._rays is None or self._rays.shape[:2] != depth.shape:
            u, v = np.meshgrid(np.arange(depth.shape[1], dtype=np.float32),
                               np.arange(depth.shape[0], dtype=np.float32))
            self._rays = np.stack([(u - intr.ppx) / intr.fx, (v - intr.ppy) / intr.fy, np.ones_like(u)],
                                  axis=-1).astype(np.float32)
        d = depth.astype(np.float32)
        points = (self._rays * d[..., None]) @ np.asarray(est.R, np.float32).T + np.asarray(est.t, np.float32)
        points[d <= 0] = np.nan
        return points

    def surface_plane(self, frame, points):
        """The work surface as a plane y = a x + b z + c (robot frame, mm): the LOWEST large
        background-coloured surface in the workspace (white blocks standing on the platform are
        white too, and can even have more depth readings than the platform). Fitted robustly,
        starting from a low percentile of the white points' heights; points far from the fit are
        dropped and it is fitted again. Falls back to a flat plane at stage_y_mm - letter_thickness_mm."""
        cfg = self.cfg
        flat = np.array([0.0, 0.0, cfg["stage_y_mm"] - cfg["letter_thickness_mm"]])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sel = (hsv_mask(hsv, cfg["background_hsv"]) > 0) & (self.workspace_mask(hsv.shape[:2]) > 0)
        P = points[sel]
        P = P[np.isfinite(P[:, 1])]
        if len(P) < 200:
            return flat
        P = P[np.random.default_rng(0).choice(len(P), min(len(P), 6000), replace=False)]
        A = np.column_stack([P[:, 0], P[:, 2], np.ones(len(P))])
        # lowest height band holding at least 15% of the white points (5 mm bands; very low
        # outliers - holes, the floor beyond the platform - hold few points each)
        ys = np.sort(P[:, 1])
        low = next((y0 for y0 in ys[::max(1, len(ys) // 200)]
                    if np.count_nonzero(np.abs(ys - (y0 + 2.5)) <= 5.0) >= 0.15 * len(ys)), float(np.median(ys)))
        coef = np.array([0.0, 0.0, low + 2.5])
        for tol in (8.0, 6.0, 5.0):
            keep = np.abs(P[:, 1] - A @ coef) < tol
            if keep.sum() < 100:
                break
            coef = np.linalg.lstsq(A[keep], P[keep, 1], rcond=None)[0]
        if np.hypot(coef[0], coef[1]) > 0.2:             # tilted > ~11 deg: not the platform, use flat
            return np.array([0.0, 0.0, low + 2.5])
        return coef

    def height_map(self, frame, depth):
        """(height of every pixel above the work surface in mm (NaN = no depth), robot-frame points)."""
        points = self.points_map(depth)
        a, b, c = self.surface_plane(frame, points)
        return points[..., 1] - (a * points[..., 0] + b * points[..., 2] + c), points

    def blob_position(self, contour, depth, plane_y, at_px=None):
        """Robot-frame position of a blob (at its centre, or at at_px, e.g. its magnet): median depth
        of the blob if enough pixels have depth, else where that pixel's ray meets the stage plane."""
        if self.camera is None:
            return None, "offline"
        mask = np.zeros(depth.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        values = depth[mask > 0]
        valid = values[values > 0]
        u, v = contour_centre(contour) if at_px is None else at_px
        if len(valid) >= 0.3 * len(values):
            return self.pixel_to_robot(u, v, np.median(valid)), "depth"
        return self.pixel_to_plane(u, v, plane_y), "plane"

    # ---------------- green tip ----------------

    def find_tip(self, frame, depth):
        """Tip pixel + robot position, or None. Only green blobs whose 3D position the arm can
        actually reach count (the camera also sees the room behind the robot), and the blob
        nearest the last known tip wins."""
        cfg = self.cfg
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.morphologyEx(hsv_mask(hsv, cfg["tip_hsv"]), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        reachable = []
        for c in contours:
            if cv2.contourArea(c) < cfg["tip_min_area_px"] or self.camera is None:
                continue
            blob = np.zeros(depth.shape, np.uint8)
            cv2.drawContours(blob, [c], -1, 255, -1)
            d = depth[blob > 0]
            d = d[d > 0]
            if not len(d):
                continue
            px = contour_centre(c)
            pos = self.pixel_to_robot(px[0], px[1], np.median(d))
            y_low, y_high = cfg["tip_y_range_mm"]
            if np.hypot(pos[0], pos[2]) <= cfg["tip_max_radius_mm"] and y_low <= pos[1] <= y_high:
                reachable.append({"px": px, "pos": pos, "contour": c})

        tip = None
        if reachable:
            if self.last_tip_px is not None:
                tip = min(reachable, key=lambda t: np.linalg.norm(t["px"] - self.last_tip_px))
                if np.linalg.norm(tip["px"] - self.last_tip_px) > cfg["tip_max_jump_px"]:
                    tip = None
            else:
                tip = max(reachable, key=lambda t: cv2.contourArea(t["contour"]))
            if tip is not None:
                self.last_tip_px = tip["px"]

        # after several misses, forget the last position and search the whole image again
        self._tip_misses = 0 if tip else self._tip_misses + 1
        if self._tip_misses >= 5:
            self.last_tip_px = None
        return tip

    def measure_tip(self, n=3, on_frame=None):
        """Median tip position over n frames; None if the tip was missed in most of them.
        Starts with a fresh search: after a commanded move the tip can be further from its last
        position than the frame-to-frame jump limit allows (that limit only applies within the n frames)."""
        self.last_tip_px = None
        pxs, poss = [], []
        for _ in range(n):
            frame, depth = self.grab()
            tip = self.find_tip(frame, depth)
            if tip is not None:
                pxs.append(tip["px"])
                poss.append(tip["pos"])
            if on_frame:
                on_frame(frame, tip)
        if len(poss) < n // 2 + 1:
            return None
        return {"px": np.median(pxs, axis=0), "pos": np.median(poss, axis=0)}

    # ---------------- letters ----------------

    def workspace_mask(self, shape):
        """Pixels whose camera ray meets the stage plane within max_radius_mm of the robot axis.
        Depends only on the camera pose, so it is computed once."""
        if self._roi is not None and self._roi.shape == shape:
            return self._roi
        if self.camera is None:
            self._roi = np.full(shape, 255, np.uint8)
            return self._roi
        est = self.camera._camera.position_estimator
        intr = est.intr
        u, v = np.meshgrid(np.arange(shape[1], dtype=float), np.arange(shape[0], dtype=float))
        rays = np.stack([(u - intr.ppx) / intr.fx, (v - intr.ppy) / intr.fy, np.ones_like(u)], axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (self.cfg["stage_y_mm"] - est.t[1]) / (rays @ est.R[1])
            points = (rays * d[..., None]) @ est.R.T + est.t
            inside = (d > 0) & (np.hypot(points[..., 0], points[..., 2]) <= self.cfg["max_radius_mm"])
        self._roi = inside.astype(np.uint8) * 255
        return self._roi

    def background_region(self, hsv):
        """Convex hull of the largest background-coloured area inside the workspace, shrunk a little."""
        background = hsv_mask(hsv, self.cfg["background_hsv"]) & self.workspace_mask(hsv.shape[:2])
        n, labels, stats, _ = cv2.connectedComponentsWithStats(background)
        if n <= 1:
            return np.zeros_like(background)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        hull = cv2.convexHull(cv2.findNonZero((labels == largest).astype(np.uint8)))
        mask = np.zeros_like(background)
        cv2.fillConvexPoly(mask, hull, 255)
        return cv2.erode(mask, np.ones((9, 9), np.uint8))

    def classify(self, contour, hsv):
        """Shape mode: returns (letter or None, score, scores per letter). Lower score = better match.
        Rotation-invariant: the blob is tried at every 10 deg against each template (and its mirror
        image, in case a letter lies face down); score = 1 - best overlap."""
        blob = normalised_masks(contour, ROTATIONS)
        scores = {L: 1.0 - best_overlap(blob, self._template_masks[L]) for L in LETTERS}
        limit = 1.0 - self.cfg["shape_min_overlap"]
        best, runner_up = sorted(scores, key=scores.get)[:2]
        clear = scores[best] <= limit and scores[runner_up] - scores[best] >= self.cfg["min_margin"]
        return (best if clear else None), scores[best], scores

    @staticmethod
    def _masks_for(contours):
        return [m for c in contours for mirror in (False, True) for m in normalised_masks(c, mirror=mirror)]

    def detect_letters(self, frame, depth):
        """Returns (detections, candidates).
        candidates: every letter-sized blob on the platform; detections: the best blob per letter."""
        if self.cfg["letter_mode"] == "color":
            return self._detect_by_color(frame, depth)
        cfg = self.cfg
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        not_background = cv2.bitwise_not(hsv_mask(hsv, cfg["background_hsv"]))
        not_robot = cv2.bitwise_not(hsv_mask(hsv, cfg["robot_hsv"]))
        mask = self.background_region(hsv) & not_background & not_robot
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if not cfg["min_area_px"] <= area <= cfg["max_area_px"]:
                continue
            letter, score, scores = self.classify(c, hsv)
            pos, source = self.blob_position(c, depth, cfg["stage_y_mm"])
            if pos is not None and np.hypot(pos[0], pos[2]) > cfg["max_radius_mm"]:
                continue
            candidates.append({"letter": letter, "score": score, "scores": scores, "contour": c,
                               "area": area, "px": contour_centre(c), "pos": pos, "source": source})

        detections = {}
        for d in candidates:
            if d["letter"] and (d["letter"] not in detections or d["score"] < detections[d["letter"]]["score"]):
                detections[d["letter"]] = d
        return detections, candidates

    def _detect_by_color(self, frame, depth):
        """Each letter has its own colour, so each colour is segmented separately. Every blob is then
        shape-checked against that letter's template, which rejects same-coloured things that aren't
        the letter (the blue arm disks for S, black cables or shadows for U)."""
        cfg = self.cfg
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = self.workspace_mask(hsv.shape[:2])
        limit = 1.0 - cfg["color_min_overlap"]
        candidates, detections = [], {}
        for L in LETTERS:
            # blobs are found on the whole image and kept if their centre is inside the search circle:
            # masking first would cut big objects (e.g. a dark computer case) into letter-sized pieces
            mask = hsv_mask(hsv, cfg["letter_hsv"][L])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            enclosed = enclosed_pixels(mask)          # holes before joining: a real magnet's hole is one
            mask = self._bridge_magnet_gaps(mask, depth)
            # RETR_CCOMP also returns holes (a magnet inside a stroke); the outer outline ignores them
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for i, c in enumerate(contours):
                if hierarchy[0][i][3] != -1:
                    continue                                 # a hole: handled with its letter
                area = cv2.contourArea(c)
                if not cfg["min_area_px"] <= area <= cfg["max_area_px"]:
                    continue
                px = contour_centre(c)
                if not roi[min(int(px[1]), roi.shape[0] - 1), min(int(px[0]), roi.shape[1] - 1)]:
                    continue
                # quick rejections, then the (slower) template check
                reason = ("round blob (arm disk?)" if ellipse_fit(c) > cfg["color_max_ellipse_fit"] else
                          "thin blob (cable or edge?)" if aspect_ratio(c) < cfg["color_min_aspect"] else None)
                score = 1.0 if reason else 1.0 - best_overlap(normalised_masks(c, ROTATIONS), self._template_masks[L])
                magnet = self._find_magnet(i, contours, hierarchy, enclosed)
                pick_px = magnet if magnet is not None else px      # go to the magnet if we can see it
                # heights come from the depth camera, so letters can be on the platform or on a block;
                # the fixed stage height is only the fallback for a letter with no depth readings
                top, support = self.blob_height(c, depth)
                plane_y = support + cfg["letter_thickness_mm"] if support is not None else cfg["stage_y_mm"]
                pos, source = self.blob_position(c, depth, plane_y, at_px=pick_px)
                if pos is not None and np.hypot(pos[0], pos[2]) > cfg["max_radius_mm"]:
                    continue
                # letters lie flat on whatever is under them; same-coloured arm parts sit far above it
                height = self._height_above_surface(top, support)
                if not reason and height is not None and height > cfg["color_max_height_mm"]:
                    reason = f"{height:.0f} mm above the surface around it (on the arm?)"
                    score = 1.0
                # ...and are flat: a same-coloured obstacle's side, seen at an angle, spans many heights
                spread = None if reason else self.blob_spread(c, depth)
                if spread is not None and spread > cfg["color_max_spread_mm"]:
                    reason = f"not flat ({spread:.0f} mm top to bottom): side of an object?"
                    score = 1.0
                d = {"letter": L if score <= limit else None, "score": score, "scores": {L: score},
                     "contour": c, "area": area, "px": pick_px, "centre_px": px, "magnet_px": magnet,
                     "pos": pos, "source": source, "color_of": L, "height_mm": height, "support_y": support,
                     "rejected": reason or (None if score <= limit else "shape doesn't match")}
                candidates.append(d)
                if d["letter"] and (L not in detections or score < detections[L]["score"]):
                    detections[L] = d
        return detections, candidates

    def blob_spread(self, contour, depth):
        """Height range (mm, 10th to 90th percentile of robot y) inside a blob, away from its edges;
        None with too few depth readings. A letter lying flat has almost none."""
        if self.camera is None:
            return None
        mask = np.zeros(depth.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        ys, xs = np.nonzero((cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0) & (depth > 0))
        if len(xs) < 30:
            return None
        y = self.pixels_to_robot(xs.astype(float), ys.astype(float), depth[ys, xs].astype(float))[:, 1]
        return float(np.percentile(y, 90) - np.percentile(y, 10))

    def _height_above_surface(self, top, support):
        """How far a blob's top is above the surface around it (mm), or None without depth.
        Without readings around the blob, the main surface (stage_y_mm - letter thickness) is used."""
        if top is None:
            return None
        surface = support if support is not None else self.cfg["stage_y_mm"] - self.cfg["letter_thickness_mm"]
        return top - surface

    # ---------------- obstacles ----------------

    def detect_obstacles(self, frame, depth, letters=(), tip=None):
        """Things the tip must go around: the given letter detections, plus every object standing at
        least obstacle_height_mm above the work surface, whatever its colour (found by height with the
        depth camera; the surface is the platform, see surface_plane, so flat marks and shadows don't
        count). The arm is removed in 3D first: seen from above, a bent arm lies on the line from the
        robot's axis to the tip (`tip`: find_tip's result), so points within arm_radius_mm of that line
        and not below the tip are the arm. Whatever still reaches more than obstacle_max_height_mm
        above the surface (a hand, the robot's frame) is dropped too.
        Each object: {"centre": (x, z), "radius", "polygon": convex footprint in the horizontal plane
        from its own 3D points, "kind": "object", "height_y": its top, "height_mm", "contour"};
        each letter: its footprint circle, with kind = the letter."""
        cfg = self.cfg
        obstacles = []
        if self.camera is not None:
            height, points = self.height_map(frame, depth)
            tall = np.nan_to_num(height, nan=-1.0) > cfg["obstacle_height_mm"]
            # pixels at a depth edge are a mix of the near and the far surface ("flying pixels"): they
            # land in mid-air between them, so ignore them
            k = np.ones((5, 5), np.uint8)
            near = cv2.erode(np.where(depth > 0, depth, 65535).astype(np.uint16), k)
            far = cv2.dilate(depth.astype(np.uint16), k)
            tall &= (far.astype(np.int32) - near.astype(np.int32)) < 15
            if tip is not None:                            # the arm: along the axis -> tip line, above the tip
                tip_pos = np.asarray(tip["pos"], float)
                ab = tip_pos[[0, 2]]
                xz = points[..., [0, 2]]
                with np.errstate(invalid="ignore"):
                    s = np.clip((xz @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
                    off_line = np.linalg.norm(xz - s[..., None] * ab, axis=-1)
                    arm = (off_line < cfg["arm_radius_mm"]) & (points[..., 1] > tip_pos[1] - 12.0)
                # + a few pixels around it: depth at the edges of the arm's disks is a mix of disk and background
                tall &= cv2.dilate(arm.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
            # letters are NOT cut out: one lying on the platform is lower than obstacle_height_mm anyway,
            # and one lying on a block is part of that block's top (cutting it out splits the block)
            # grey/white and coloured things are grouped separately: a (shaded) white block next to a
            # coloured obstacle (touching in the image, at the same height where they meet) stays two
            white = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1] <= cfg["obstacle_grey_max_saturation"]
            letter_mask = np.zeros(tall.shape, np.uint8)       # a (coloured) letter on a block joins that block
            for d in letters:
                cv2.drawContours(letter_mask, [d["contour"]], -1, 255, -1)
            white |= cv2.dilate(letter_mask, np.ones((7, 7), np.uint8)) > 0
            roi = self.workspace_mask(tall.shape) > 0
            blobs = []
            for part in (tall & white, tall & ~white):
                mask = part.astype(np.uint8) * 255
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # bridge depth holes
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))    # drop specks, edge noise
                n, labels = cv2.connectedComponents(mask)
                blobs += [labels == k for k in range(1, n)]
            for blob in blobs:
                here = blob & roi
                if np.count_nonzero(here) < cfg["obstacle_min_area_px"]:
                    continue
                h = height[here]
                h = h[np.isfinite(h)]
                if len(h) < 30 or np.percentile(h, 98) > cfg["obstacle_max_height_mm"]:
                    continue                                   # reaches far up: a hand, the robot's frame
                core = cv2.erode(here.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0   # skip edge pixels
                P = points[core]
                P = P[np.isfinite(P[:, 0])]
                if len(P) < 20:
                    continue
                polygon = convex_hull(P[:, [0, 2]])
                if len(polygon) < 3:
                    continue
                centre = polygon.mean(axis=0)
                contours, _ = cv2.findContours(here.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hc = height[core]
                ok = np.isfinite(hc) & np.isfinite(points[core][:, 1])
                base = points[core][ok, 1] - hc[ok] if ok.any() else None   # surface height under it
                obstacles.append({"centre": centre, "radius": float(np.linalg.norm(polygon - centre, axis=1).max()),
                                  "polygon": polygon, "kind": "object",
                                  "base_y": float(np.median(base)) if base is not None else None,
                                  "height_y": float(np.percentile(P[:, 1], 90)),
                                  "height_mm": float(np.percentile(h, 90)),
                                  "contour": max(contours, key=cv2.contourArea)})
        for d in letters:
            footprint = self._footprint(d["contour"], None if d.get("pos") is None else d["pos"][1])
            if footprint is not None:
                obstacles.append({**footprint, "kind": d.get("letter") or d.get("color_of") or "?",
                                  "contour": d["contour"]})
        return obstacles

    def _footprint(self, contour, top_y):
        """Circle (centre (x, z), radius mm) around a blob's outline projected onto its own height."""
        if self.camera is None:
            return None
        y = top_y if top_y is not None else self.cfg["stage_y_mm"]
        points = [self.pixel_to_plane(float(u), float(v), y) for u, v in contour.reshape(-1, 2)[::3]]
        points = np.array([p[[0, 2]] for p in points if p is not None])
        if len(points) < 3:
            return None
        centre = points.mean(axis=0)
        return {"centre": centre, "radius": float(np.linalg.norm(points - centre, axis=1).max()), "height_y": y}

    def _bridge_magnet_gaps(self, mask, depth):
        """A letter's magnet is lighter than its paint, and when it touches the stroke's edge it cuts
        the letter into pieces. Join pieces whose ends are within magnet_gap_px by filling only the
        gap between their facing ends (convex hull of those ends), so the letter's shape is otherwise
        untouched. Round pieces (arm disks) and pieces above the surface (on the arm) are never joined."""
        cfg = self.cfg
        gap = cfg["magnet_gap_px"]
        n, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
        pieces = []
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < 30:
                continue
            x, y, w, h = stats[k, :4]
            outline, _ = cv2.findContours((labels[y:y + h, x:x + w] == k).astype(np.uint8),
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outline = max(outline, key=cv2.contourArea) + [x, y]
            if ellipse_fit(outline) > cfg["color_max_ellipse_fit"]:
                continue                                   # round: an arm disk, not part of a letter
            height = self._height_above_surface(*self.blob_height(outline, depth))
            if height is not None and height > cfg["color_max_height_mm"]:
                continue                                   # far above the surface around it: on the arm
            pieces.append((stats[k, :4], outline.reshape(-1, 2)))

        for (box_a, pts_a), (box_b, pts_b) in combinations(pieces, 2):
            xa, ya, wa, ha = box_a
            xb, yb, wb, hb = box_b
            if xa > xb + wb + gap or xb > xa + wa + gap or ya > yb + hb + gap or yb > ya + ha + gap:
                continue                                   # bounding boxes too far apart
            dist = np.linalg.norm(pts_a[:, None, :].astype(float) - pts_b[None, :, :], axis=2)
            if dist.min() > gap:
                continue
            # fill only around the cut (the closest point pair), about one magnet radius: an S's two
            # halves are also close along its inner curves, which must stay open
            ia, ib = np.unravel_index(np.argmin(dist), dist.shape)
            cut = (pts_a[ia] + pts_b[ib]) / 2.0
            radius = max(dist.min() / 2.0, gap / 2.0) + 2
            ends = np.vstack([pts_a[np.linalg.norm(pts_a - cut, axis=1) <= radius],
                              pts_b[np.linalg.norm(pts_b - cut, axis=1) <= radius]]).astype(np.int32)
            cv2.fillConvexPoly(mask, cv2.convexHull(ends), 255)
        return mask

    def _find_magnet(self, i, contours, hierarchy, enclosed):
        """Pixel of letter i's magnet: its largest round, magnet-sized hole that already existed
        before joining (joining can close off pockets of an S's inner curves), or None. When the
        magnet only bites a notch into the edge there's no hole; N's and S's magnets are at the
        letter's centre, so the caller then uses the centre. U's magnet sits inside the stroke."""
        low, high = self.cfg["magnet_area_px"]
        holes = []
        child = hierarchy[0][i][2]
        while child != -1:
            hole = contours[child]
            area = cv2.contourArea(hole)
            circularity = 4 * np.pi * area / max(cv2.arcLength(hole, True) ** 2, 1e-6)
            if low <= area <= high and circularity >= self.cfg["magnet_min_circularity"]:
                centre = contour_centre(hole)
                if enclosed[min(int(round(centre[1])), enclosed.shape[0] - 1),
                            min(int(round(centre[0])), enclosed.shape[1] - 1)]:
                    holes.append((area, centre))
            child = hierarchy[0][child][0]
        return max(holes, key=lambda h: h[0])[1] if holes else None

    def learn_color(self, letter, frame, x, y, tolerance=30, window=60):
        """Learn a letter's colour from the paint at (x, y) and save it to nus_config.json.
        Flood-fills the connected pixels of similar colour (like a paint bucket), so only the
        letter's own paint is used, not the platform around it. Returns the ranges, or None."""
        h, w = frame.shape[:2]
        x0, y0 = max(0, x - window), max(0, y - window)
        crop = frame[y0:min(h, y + window + 1), x0:min(w, x + window + 1)]
        smooth = cv2.medianBlur(crop, 5)                 # so one noisy seed pixel can't spoil the fill
        fill = np.zeros((crop.shape[0] + 2, crop.shape[1] + 2), np.uint8)
        cv2.floodFill(smooth, fill, (x - x0, y - y0), 0, (tolerance,) * 3, (tolerance,) * 3,
                      4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[fill[1:-1, 1:-1] > 0]
        if len(pixels) < 20:
            return None
        ranges = learn_hsv_range(pixels.reshape(-1, 1, 3))
        self.cfg["letter_hsv"][letter] = ranges
        save_config_value("letter_hsv", self.cfg["letter_hsv"])
        return ranges

    def teach(self, letter, contour, replace=False):
        """Store a real letter outline as a template. replace=True first forgets that letter's
        earlier examples (e.g. ones taught before the magnet gaps were joined up)."""
        if replace:
            self.taught[letter] = []
        self.taught.setdefault(letter, []).append(contour)
        self.templates[letter] = (default_templates()[letter] + self.taught[letter] if self.color_mode
                                  else self.taught[letter])
        self._template_masks[letter] = self._masks_for(self.templates[letter])
        np.savez(self.template_path, **{f"{L}_{i}": c for L, cs in self.taught.items() for i, c in enumerate(cs)})


def draw_overlay(frame, candidates=(), detections=None, tip=None, target_px=None, lines=(), roi=None):
    out = frame.copy()
    if roi is not None and np.count_nonzero(roi) < roi.size:      # outline of the searched workspace
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (200, 200, 200), 1)
    chosen = {id(d) for d in (detections or {}).values()}
    for d in candidates:
        colour = (0, 200, 0) if id(d) in chosen else (0, 140, 255) if d["letter"] else (0, 0, 255)
        cv2.drawContours(out, [d["contour"]], -1, colour, 2)
        u, v = (int(x) for x in d["px"])
        # colour mode: a rejected blob shows which letter's colour it had, e.g. "S? 0.71"
        name = d["letter"] or (f'{d["color_of"]}?' if d.get("color_of") else "?")
        cv2.putText(out, f'{name} {d["score"]:.2f}', (u + 10, v), FONT, 0.5, colour, 2)
        if d.get("magnet_px") is not None:                           # the point the robot goes to
            cv2.circle(out, (u, v), 5, (255, 255, 255), -1)
            cv2.circle(out, (u, v), 5, colour, 2)
        if d["pos"] is not None:
            cv2.putText(out, f'{fmt(d["pos"])} [{d["source"]}]', (u + 10, v + 16), FONT, 0.4, colour, 1)
    if tip is not None:
        cv2.circle(out, tuple(int(x) for x in tip["px"]), 8, (0, 255, 255), 2)
    if target_px is not None:
        cv2.drawMarker(out, tuple(int(x) for x in target_px), (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
    for i, text in enumerate(lines):
        y = 20 + 18 * i
        cv2.putText(out, text, (10, y), FONT, 0.5, (0, 0, 0), 3)
        cv2.putText(out, text, (10, y), FONT, 0.5, (255, 255, 255), 1)
    return out
