"""
Camera window shared by the nus_main commands: live overlay (letters, tip, obstacles, planned
routes, marks), an optional MP4 recorder, and helpers.
"""

import sys
import time

import cv2
import numpy as np

from nus_path import footprint
from nus_vision import LETTERS, Vision, draw_overlay

WINDOW = "NUS retrieval  (q / Esc = stop)"


class Display:
    def __init__(self, vision, recorder=None):
        self.vision = vision
        self.recorder = recorder
        self.candidates, self.detections = [], {}
        self.target_px, self.lines = None, []
        self.target_pos, self.timer_start = None, None     # live distance + timer while moving
        self.obstacles, self.clearance_mm = [], 0.0          # drawn as outlines (footprint + clearance)
        self.path = None                                     # (list of (x, z), height y): planned route
        self.alt_paths = []                                  # other candidate routes, drawn thin
        self.marks = []                                      # (robot (x, y, z), label, BGR colour)
        self.region = None                                   # (outline [(x, z)], height y): where the tip reaches
        cv2.namedWindow(WINDOW)

    def _to_px(self, x, y, z):
        p = self.vision.robot_to_pixel([x, y, z])
        return None if p is None else tuple(int(v) for v in p)

    def _draw_plan(self, image):
        for o in self.obstacles:
            if o["kind"] == "object" and o.get("contour") is not None:   # what the camera segmented
                cv2.drawContours(image, [o["contour"]], -1, (0, 100, 255), 1, cv2.LINE_AA)
            # footprint (thick) and the offset the tip keeps from it (thin), drawn where it stands
            y = o["base_y"] if o.get("base_y") is not None else o["height_y"]
            for offset, thickness in ((0.0, 2), (o.get("inflate", self.clearance_mm), 1)):
                ring = [self._to_px(x, y, z) for x, z in footprint(o, offset)]
                ring = np.array([p for p in ring if p is not None], np.int32)
                if len(ring) > 2:
                    cv2.polylines(image, [ring], True, (0, 165, 255), thickness, cv2.LINE_AA)
            if o["kind"] == "object" and "height_mm" in o:
                p = self._to_px(o["centre"][0], o["height_y"], o["centre"][1])
                if p is not None:
                    cv2.putText(image, f"{o['height_mm']:.0f}mm", p, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 165, 255), 1, cv2.LINE_AA)
        if self.region is not None:
            outline, y = self.region
            ring = np.array([p for p in (self._to_px(x, y, z) for x, z in outline) if p is not None], np.int32)
            if len(ring) > 2:
                cv2.polylines(image, [ring], True, (80, 200, 80), 1, cv2.LINE_AA)
        for points, y in self.alt_paths:
            line = np.array([p for p in (self._to_px(x, y, z) for x, z in points) if p is not None], np.int32)
            if len(line) > 1:
                cv2.polylines(image, [line], False, (170, 170, 170), 1, cv2.LINE_AA)
        for point, label, colour in self.marks:
            p = self._to_px(*point)
            if p is not None:
                cv2.circle(image, p, 6, colour, 2, cv2.LINE_AA)
                cv2.putText(image, label, (p[0] + 8, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        if self.path is not None:
            points, y = self.path
            line = np.array([p for p in (self._to_px(x, y, z) for x, z in points) if p is not None], np.int32)
            if len(line) > 1:
                cv2.polylines(image, [line], False, (255, 255, 0), 2, cv2.LINE_AA)

    def show(self, frame, tip=None):
        roi = self.vision.workspace_mask(frame.shape[:2])
        lines = list(self.lines)
        if self.timer_start is not None:
            lines.append(f"time {time.time() - self.timer_start:5.1f} s")
        if self.target_pos is not None and tip is not None:
            lines.append(f"distance to target {np.hypot(*(tip['pos'][[0, 2]] - self.target_pos[[0, 2]])):5.1f} mm")
        image = draw_overlay(frame, self.candidates, self.detections, tip, self.target_px, lines, roi)
        self._draw_plan(image)
        if self.target_px is not None and tip is not None and self.path is None:
            cv2.line(image, tuple(int(v) for v in tip["px"]), tuple(int(v) for v in self.target_px),
                     (255, 0, 255), 1, cv2.LINE_AA)
        if self.recorder is not None:
            self.recorder.write(image)
        cv2.imshow(WINDOW, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt("stopped from the camera window")
        return key


class Recorder:
    """Saves the annotated camera view as a real-time MP4 (frames are repeated or skipped to keep
    a steady frame rate), e.g. for the project video."""

    def __init__(self, path, fps=15):
        self.path, self.fps = path, fps
        self.writer, self.start, self.written = None, None, 0

    def write(self, image):
        if self.writer is None:
            self.path.parent.mkdir(exist_ok=True)
            h, w = image.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.start = time.time()
        due = int((time.time() - self.start) * self.fps) + 1
        while self.written < due:
            self.writer.write(image)
            self.written += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            print(f"Video saved: {self.path}")


def open_camera_or_exit():
    camera = Vision.open_camera()
    if camera is None:
        sys.exit("Could not open the camera. Close EMIO Labs and check the USB cable.")
    return camera


def locate_letters(vision, display, frames=15):
    """Median position of each letter over several frames (more stable than one frame)."""
    seen = {L: [] for L in LETTERS}
    for _ in range(frames):
        frame, depth = vision.grab()
        detections, candidates = vision.detect_letters(frame, depth)
        display.candidates, display.detections = candidates, detections
        display.show(frame)
        for L, d in detections.items():
            if d["pos"] is not None:
                seen[L].append(d)
    found = {}
    for L, ds in seen.items():
        if len(ds) >= frames // 3:
            found[L] = {"pos": np.median([d["pos"] for d in ds], axis=0),
                        "px": np.median([d["px"] for d in ds], axis=0),
                        "score": float(np.median([d["score"] for d in ds]))}
    return found


def stable_objects(history, min_frames=3, match_mm=15.0):
    """Objects of the latest frame that were also seen (centre within match_mm) in at least
    min_frames of the recent frames: depth noise can make a spurious blob appear for a frame."""
    if not history:
        return []
    return [o for o in history[-1]
            if sum(any(np.linalg.norm(np.asarray(o["centre"]) - np.asarray(p["centre"])) < match_mm for p in frame)
                   for frame in history) >= min(min_frames, len(history))]
