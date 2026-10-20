"""
N-U-S letter retrieval: find the letters with the camera and steer the manipulator tip over them.

Run inside the ME4242 conda env, with EMIO Labs closed:
    python nus_main.py check --apply       # camera only: check a (new) set, save its surface height
    python nus_main.py detect              # camera only: check letter detection, teach letter shapes
    python nus_main.py jog                 # wind cables by keyboard: take up slack, find pull directions
    python nus_main.py calibrate           # find each cable's pull direction + taut point, measure the model
    python nus_main.py calibrate --pull-signs=-1,-1,1,1   # pull directions known (e.g. from jog)
    python nus_main.py goto N              # steer the tip over letter N  (or U, S, all)
    python nus_main.py goto all --dry-run  # detect and plan only, motors never move
    python nus_main.py search              # N -> U -> S in turn, timed, results saved to runs/
    python nus_main.py search --trials 5 --record   # 5 trials, saves a video of each session

Press q or Esc in the camera window at any time to stop. Stopping (or finishing)
disables motor torque, so the arm goes slack.

Steps 1-3 of the task (find letter, raise the tip, move over it, lower onto it) are covered here.
Carrying the letter past the obstacles and placing it in the mold come next.
"""

import argparse
import csv
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from nus_control import (MAX_TRAVEL_RAD, PULL_MAX_MM, PULLEY_RADIUS_MM, Cables, align_to_calibration, calibrate,
                         describe_model, load_calibration, lower_onto, move_tip_adaptive, move_tip_along, move_tip_to,
                         probe_model, raise_tip, reachability, reset_live_model, set_live_model, solve_pulls)
from nus_path import densify, footprint, min_clearance, path_length, plan_path, touched
from nus_vision import LETTERS, Vision, draw_overlay, fmt, load_config, save_config_value

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


# ---------------------------------------------------------------- detect

def teach_color_and_shape(vision, letter, frame, depth, click, replace=False):
    """Colour mode: learn the letter's colour from the paint at the click, then segment with that
    colour and store the blob there as a shape example (replace=True forgets older examples)."""
    x, y = int(click[0]), int(click[1])
    ranges = vision.learn_color(letter, frame, x, y)
    if ranges is None:
        print(f"  Couldn't learn {letter}'s colour: click ON the letter's paint (a stroke, not a gap).")
        return
    print(f"Learned {letter}'s colour: HSV ranges {ranges} (saved to nus_config.json)")
    _, candidates = vision.detect_letters(frame, depth)
    same_colour = [d for d in candidates if d.get("color_of") == letter]
    # nearest blob of that colour (signed distance: > 0 inside, < 0 outside), so a click in a gap works too
    distance = {id(d): cv2.pointPolygonTest(d["contour"], (float(x), float(y)), True) for d in same_colour}
    blob = max(same_colour, key=lambda d: distance[id(d)], default=None)
    if blob is None or distance[id(blob)] < -30:
        print(f"  No {letter}-coloured blob at the click, so the shape was not stored. Click ON the letter's "
              "paint (a stroke, not a gap) and press the key again.")
        return
    vision.teach(letter, blob["contour"], replace=replace)
    magnet = "magnet found" if blob.get("magnet_px") is not None else "no magnet seen"
    print(f"  ...and its shape ({len(vision.taught[letter])} example(s) saved"
          f"{', older ones replaced' if replace else ''}; {magnet}).")
    return True


def cmd_detect(cfg, args):
    camera = open_camera_or_exit()
    vision = Vision(cfg, camera)
    display = Display(vision)
    state = {"click": None, "frame": None, "depth": None, "candidates": [], "taught": set()}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or state["frame"] is None:
            return
        state["click"] = np.array([x, y])
        pixel = np.ascontiguousarray(state["frame"][y:y + 1, x:x + 1])
        h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
        above = vision.height_map(state["frame"], state["depth"])[0][y, x]
        print(f"pixel ({x}, {y}): HSV = ({h}, {s}, {v}) | depth = {int(state['depth'][y, x])} mm | "
              f"on the stage plane: {fmt(vision.pixel_to_plane(x, y, cfg['stage_y_mm']))} | "
              + (f"{above:.0f} mm above the work surface" if np.isfinite(above) else "no depth reading"))

    cv2.setMouseCallback(WINDOW, on_mouse)
    teaches = "its colour and shape" if cfg["letter_mode"] == "color" else "its shape"
    print(f"Click a pixel to print its colour and position. Click ON a letter's paint (a stroke, not a "
          f"gap), then press n / u / s to teach {teaches}. w saves a snapshot, q quits.")
    try:
        while True:
            frame, depth = vision.grab()
            detections, candidates = vision.detect_letters(frame, depth)
            tip = vision.find_tip(frame, depth)
            objects = [o for o in vision.detect_obstacles(frame, depth, list(detections.values()), tip=tip)
                       if o["kind"] == "object"]
            state.update(frame=frame, depth=depth, candidates=candidates)
            display.candidates, display.detections, display.obstacles = candidates, detections, objects
            display.lines = [f"{L}: {fmt(detections[L]['pos'])}  score {detections[L]['score']:.2f}"
                             if L in detections else f"{L}: not found" for L in LETTERS]
            display.lines.append(f"mode: {cfg['letter_mode']} | {len(candidates)} candidate blob(s)")
            display.lines.append(f"obstacles (orange): {len(objects)}" + (" - " + ", ".join(
                f"{o['height_mm']:.0f} mm tall" for o in objects if 'height_mm' in o) if objects else ""))
            key = display.show(frame, tip)

            if key in (ord("n"), ord("u"), ord("s")):
                letter = chr(key).upper()
                if state["click"] is None:
                    print("Click on the middle of the letter first.")
                    continue
                if cfg["letter_mode"] == "color":
                    # the first example of a letter in this session replaces its old ones
                    if teach_color_and_shape(vision, letter, frame, depth, state["click"],
                                             replace=letter not in state["taught"]):
                        state["taught"].add(letter)
                    continue
                if not candidates:
                    print("No letter-sized blob found: click on the letter first.")
                    continue
                nearest = min(candidates, key=lambda d: np.linalg.norm(d["px"] - state["click"]))
                vision.teach(letter, nearest["contour"])
                print(f"Taught {letter} ({len(vision.taught[letter])} example(s) saved).")
            elif key == ord("w"):
                name = f"snapshot_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(name, frame)
                print(f"Saved {name}")
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------- check (camera only)

def cmd_check(cfg, args):
    """Quick check of a (new) EMIO set before calibrating. The motors are never moved."""
    from emioapi._motorgroup import listMotors
    print("\n=== Setup check (motors are not moved) ===")
    ports = listMotors()
    print(f"Motor board : {ports[0] if ports else 'NOT FOUND - check the USB cable and power'}")
    calib = load_calibration()
    print("Cable calib : " + (f"from {calib['created']} - run calibrate again on a new set" if calib else "none yet"))
    camera = Vision.open_camera()
    if camera is None:
        print("Camera      : NOT FOUND - close EMIO Labs, check the USB cable")
        return
    vision = Vision(cfg, camera)
    display = Display(vision)
    try:
        for _ in range(15):                                  # let auto-exposure settle
            frame, depth = vision.grab()
            display.show(frame)

        tip = vision.measure_tip(10, display.show)
        if tip is None:
            print("Tip         : NOT FOUND - is the green tip in view? (only green blobs near the robot count)")
        else:
            off = np.hypot(tip["pos"][0], tip["pos"][2])
            verdict = ("OK" if off <= 15 else
                       "off-centre: expected if the arm rests bent on the raised platform. If it hangs "
                       "straight instead, this set's camera may need its own calibration "
                       "(python -m emioapi calibrate, see aruco_672_calibration_marker.pdf)")
            print(f"Tip         : {fmt(tip['pos'])}, {off:.0f} mm off the robot axis -> {verdict}")

        heights = []
        for _ in range(5):
            frame, depth = vision.grab()
            h, _ = vision.surface_height(frame, depth)
            if h is not None:
                heights.append(h)
        if not heights:
            print("Surface     : could not measure (no dark cloth with depth readings near the centre?)")
        else:
            surface = float(np.median(heights))
            suggested = round(surface + args.letter_mm)
            gap = f", tip hangs {tip['pos'][1] - surface:.0f} mm above it" if tip is not None else ""
            print(f"Surface     : y = {surface:.0f} mm{gap}")
            print(f"stage_y_mm  : currently {cfg['stage_y_mm']:.0f}, suggested {suggested} "
                  f"(surface + {args.letter_mm:.0f} mm letter thickness)")
            if args.apply:
                save_config_value("stage_y_mm", float(suggested))
                cfg["stage_y_mm"] = float(suggested)
                vision._roi = None                           # search circle depends on this height
                print(f"              -> saved to nus_config.json")
            elif abs(suggested - cfg["stage_y_mm"]) > 5:
                print("              -> run `python nus_main.py check --apply` to save it")

        found = locate_letters(vision, display)
        for L in LETTERS:
            print(f"Letter {L}    : " + (f"found at {fmt(found[L]['pos'])}" if L in found
                                         else "not found (in the circle? taught? see `detect`)"))
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------- calibrate

def cmd_calibrate(cfg, args):
    how = (f"in the given pull direction {args.pull_signs}" if args.pull_signs
           else "in each direction until it finds which way pulls")
    print(f"Calibration turns each motor up to {args.search_mm:.0f} mm of cable, one at a time, {how},\n"
          "and stops as soon as the cable becomes taut. Leave the arm at rest (straight, or lying bent\n"
          "on the raised platform), with no cable pulling.\n"
          "Keep hands clear. Press q in the camera window to stop at any time.")
    if input("Start? [y/N] ").strip().lower() != "y":
        return
    camera = open_camera_or_exit()
    cables = Cables()
    vision = Vision(cfg, camera)
    display = Display(vision)
    display.lines = ["calibrating..."]
    try:
        if not cables.open():
            print("Could not connect to the motors (power on? USB plugged in? EMIO Labs closed?).")
            return
        calibrate(cables, vision, display.show, first_direction=args.first_direction,
                  pull_signs=args.pull_signs, search_mm=args.search_mm)
        print("Calibration done.")
    except KeyboardInterrupt:
        print("Stopped.")
    except (RuntimeError, ValueError) as e:
        print(f"Calibration failed: {e}")
    finally:
        cables.close()
        camera.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------- search: N -> U -> S

SEQUENCE = ("N", "U", "S")
RUNS_DIR = Path(__file__).resolve().parent / "runs"
RESULTS_CSV = RUNS_DIR / "search_results.csv"
CSV_FIELDS = ["time", "trial", "letter", "success", "error_mm", "seconds", "steps", "reason",
              "target_x", "target_y", "target_z", "tip_x", "tip_y", "tip_z", "success_mm",
              "waypoints", "path_mm", "min_clearance_mm", "raised_mm", "above_letter_mm"]
DEFAULT_LETTER_RADIUS_MM = 25.0
STUCK_MM_PER_MM = 0.5                 # no cable moves the tip more than this per mm of cable: it's caught      # footprint used for a letter that wasn't in the last frame


def ready_screen(vision, display, calib, tip_rest, args, trial):
    """Live letter and obstacle detection until SPACE. Shows which letters are found, whether each
    looks reachable (estimated from the calibrated model) and the obstacles the tip will avoid.
    Returns ({letter: {"pos", "px"}}, obstacles) where obstacles includes every found letter
    (kind = its letter) and other objects (kind = "object")."""
    J = np.array(calib["J"])[[0, 2], :]
    history = {L: deque(maxlen=15) for L in SEQUENCE}
    display.target_px = display.target_pos = display.timer_start = display.path = None
    print(f"\nTrial {trial}/{args.trials}: place the letters, then press SPACE in the camera window "
          "(f = start with the letters found so far, q = quit).")
    object_history = deque(maxlen=5)
    all_found_since = None
    while True:
        frame, depth = vision.grab()
        detections, candidates = vision.detect_letters(frame, depth)
        tip = vision.find_tip(frame, depth)
        obstacles = [] if args.route == "straight" else vision.detect_obstacles(
            frame, depth, list(detections.values()), tip=tip)
        object_history.append([o for o in obstacles if o["kind"] == "object"])
        obstacles = stable_objects(object_history) + [o for o in obstacles if o["kind"] != "object"]
        display.obstacles = with_offsets(obstacles, args)
        for L in SEQUENCE:
            d = detections.get(L)
            history[L].append(d if d is not None and d["pos"] is not None else None)

        found = {}
        lines = [f"Trial {trial}/{args.trials}   SPACE = start   f = start anyway   q = quit"]
        for L in SEQUENCE:
            seen = [d for d in history[L] if d is not None]
            if len(seen) < 8:                               # seen in at least 8 of the last 15 frames
                lines.append(f"{L}: not found")
                continue
            pos = np.median([d["pos"] for d in seen], axis=0)
            offset = pos[[0, 2]] - tip_rest[[0, 2]]
            _, short = solve_pulls(J, offset, calib["pull_max_mm"])
            found[L] = {"pos": pos, "px": np.median([d["px"] for d in seen], axis=0)}
            lines.append(f"{L}: {np.hypot(*offset):3.0f} mm from the tip, height y {pos[1]:.0f}, " +
                         ("looks reachable" if short <= args.success_mm
                          else f"probably OUT OF REACH ({short:.0f} mm short)"))
        objects = [o for o in obstacles if o["kind"] == "object"]
        if args.route != "straight":
            lines.append(f"obstacles: {len(objects)}" + (" (" + ", ".join(
                f"{o['height_mm']:.0f} mm tall" for o in objects if 'height_mm' in o) + ")" if objects else ""))
        if args.route == "contact":
            lines.append(f"route: lean on {len(objects)} object(s) (offset {args.hug_mm:.0f} mm), keep {args.clearance_mm:.0f} mm from letters")
        elif args.route == "avoid":
            lines.append(f"route: keep {args.clearance_mm:.0f} mm from letters and {len(objects)} object(s)")
        if args.raise_tip:
            lines.append("raise the tip first (bend it up), move toward the letter raised, then come down onto it")
        display.candidates, display.detections, display.lines = candidates, detections, lines
        key = display.show(frame, tip)
        all_found_since = (all_found_since or time.time()) if len(found) == len(SEQUENCE) else None
        auto = args.auto_start is not None and all_found_since is not None and time.time() - all_found_since >= args.auto_start
        if auto or (key == ord(" ") and len(found) == len(SEQUENCE)) or (key == ord("f") and found):
            letter_obstacles = {o["kind"]: o for o in obstacles if o["kind"] in found}
            for L, f in found.items():           # a found letter missing from this last frame
                letter_obstacles.setdefault(L, {"centre": f["pos"][[0, 2]], "radius": DEFAULT_LETTER_RADIUS_MM,
                                                "height_y": f["pos"][1], "kind": L})
            return found, objects + list(letter_obstacles.values())
        if key == ord(" "):
            print("Not all letters found yet. Press f to start with: " + (", ".join(found) or "none"))


def stable_objects(history, min_frames=3, match_mm=15.0):
    """Objects of the latest frame that were also seen (centre within match_mm) in at least
    min_frames of the recent frames: depth noise can make a spurious blob appear for a frame."""
    if not history:
        return []
    return [o for o in history[-1]
            if sum(any(np.linalg.norm(np.asarray(o["centre"]) - np.asarray(p["centre"])) < match_mm for p in frame)
                   for frame in history) >= min(min_frames, len(history))]


def save_and_summarise(rows):
    RUNS_DIR.mkdir(exist_ok=True)
    path = RESULTS_CSV
    if path.exists():
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != CSV_FIELDS:              # written by an older version with other columns
            path = RUNS_DIR / f"search_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)

    print("\n================ Summary of this session ================")
    for L in SEQUENCE:
        attempts = [r for r in rows if r["letter"] == L]
        reached = [r for r in attempts if r["success"]]
        if attempts:
            extra = (f" | mean {np.mean([r['seconds'] for r in reached]):.1f} s, "
                     f"mean final error {np.mean([r['error_mm'] for r in reached]):.1f} mm") if reached else ""
            print(f"  {L}: reached {len(reached)}/{len(attempts)} ({100 * len(reached) / len(attempts):.0f}%){extra}")
    trials = [r for r in rows if r["letter"] == "ALL"]
    if trials:
        complete = [r for r in trials if r["success"]]
        print(f"  Full N-U-S sequence: {len(complete)}/{len(trials)} trials | "
              f"mean time {np.mean([r['seconds'] for r in trials]):.1f} s")
    print(f"  Results appended to {path}")


def with_offsets(obstacles, args):
    """Offset per obstacle: letters keep a clearance (so they aren't dragged); in contact mode,
    rigid objects only get the arm's radius, so the route wraps round them and the arm lies on them."""
    hug = args.route == "contact"
    return [dict(o, inflate=args.hug_mm if (hug and o["kind"] == "object") else args.clearance_mm)
            for o in obstacles]


def plan_route(vision, start_xz, target, letter, obstacles, reachable, args):
    """Path for the tip from start_xz to the target letter. "contact": wraps round rigid objects
    (they're pivots that give the arm its second bend) and keeps clear of other letters; "avoid":
    keeps clear of everything; "straight": no planning.
    Returns (waypoints for move_tip_along, full path for drawing, obstacles on the route's way)."""
    goal = np.asarray(target, float)[[0, 2]]
    start = np.asarray(start_xz, float)
    # everything but the target, minus anything the tip is on now (the letter it just visited)
    others = with_offsets([o for o in obstacles if o["kind"] != letter
                           and all(np.linalg.norm(p - np.asarray(o["centre"], float)) > o["radius"]
                                   for p in (start, goal))], args)
    if args.route == "straight" or not others:
        return [goal], [start, goal], others
    # free-space reach doesn't apply when the arm can lean on an object: don't limit waypoints then
    limit = None if args.route == "contact" and any(o["kind"] == "object" for o in others) else reachable
    path = plan_path(start, goal, others, args.clearance_mm, limit)
    if path is None:
        print("  No route found around the obstacles: going straight.")
        path = [start, goal]
    # follow the path closely (even a straight one), so the arm doesn't cut corners
    return densify(path, args.waypoint_step_mm), path, others


def cmd_search(cfg, args):
    calib = load_calibration()
    if calib is None:
        sys.exit("No cable_calibration.json yet: run `python nus_main.py calibrate` first.")
    camera = open_camera_or_exit()
    vision = Vision(cfg, camera)
    recorder = Recorder(RUNS_DIR / f"search_{time.strftime('%Y%m%d_%H%M%S')}.mp4") if args.record else None
    display = Display(vision, recorder)
    cables = Cables(calib)
    rows = []
    try:
        if not cables.open():
            print("Could not connect to the motors (power on? USB plugged in? EMIO Labs closed?).")
            return
        align_to_calibration(cables)            # refuses if a spool moved since calibration
        if args.pull_max_mm is not None:
            cables.calib = {**cables.calib, "pull_max_mm": [args.pull_max_mm if s else 0.0
                                                            for s in cables.calib["pull_sign"]]}
        cables.set_pulls(np.zeros(4))
        tip_rest = vision.measure_tip(10, display.show)
        if tip_rest is None:
            print("Can't see the green tip. Check the camera view.")
            return

        reachable = reachability(cables.calib, tip_rest["pos"])
        for trial in range(1, args.trials + 1):
            found, obstacles = ready_screen(vision, display, cables.calib, tip_rest["pos"], args, trial)
            display.candidates, display.detections = [], {}
            status = {L: ".." for L in SEQUENCE}
            reset_live_model(cables)                        # learned model starts from the calibration
            trial_start = display.timer_start = time.time()

            for L in SEQUENCE:
                row = {"time": datetime.now().isoformat(timespec="seconds"), "trial": trial, "letter": L,
                       "success_mm": args.success_mm}
                if L not in found:
                    print(f"[trial {trial}] {L}: not detected, skipped")
                    rows.append({**row, "success": 0, "reason": "not detected"})
                    status[L] = "skip"
                    continue
                target = found[L]["pos"]
                display.target_px, display.target_pos = found[L]["px"], target
                status[L] = "->"
                status_line = "   ".join(f"{k} {v}" for k, v in status.items()) + f"   (trial {trial})"
                print(f"\n[trial {trial}] going to {L} at {fmt(target)}")

                t0 = time.time()
                # 0. measure how each cable moves the tip from here: the calibration's model is only
                #    right near the rest pose, and after the last letter the arm is bent differently
                display.lines = [status_line, f"{L}: measuring the cables"]
                J, now = probe_model(cables, vision, display.show)
                if J is not None and np.linalg.norm(J[[0, 2]], axis=0).max() < STUCK_MM_PER_MM and cables.pulls.any():
                    # no cable moves the tip: it is caught on something (e.g. a block's edge, with the
                    # cables under tension). Let the cables go back to rest so the arm springs free.
                    print("  the tip is stuck (no cable moves it): letting the cables go back to rest first")
                    cables.set_pulls(np.zeros(4))
                    J, now = probe_model(cables, vision, display.show)
                if J is not None:
                    set_live_model(cables, J)
                    print("  cables here: " + describe_model(J, cables.used > 0))
                # 1. bend the tip up, only with cables that also move it toward the letter
                raised, hold_y = {"raised_mm": 0.0}, None
                if args.raise_tip and J is not None:
                    display.lines = [status_line, f"{L}: raising the tip"]
                    raised = raise_tip(cables, vision, display.show, J=J, max_load_ma=args.raise_max_ma,
                                       stop_at_y=target[1] + args.raise_above_mm if args.raise_above_mm > 0 else None,
                                       toward_xz=target[[0, 2]])
                    now = raised["tip"] or now
                    if now is not None and raised["raised_mm"] >= 2.0:
                        hold_y = float(now["pos"][1])
                    print(f"  raised the tip {max(raised['raised_mm'], 0):.0f} mm ({raised['reason']})"
                          + (f"; it is now {now['pos'][1] - target[1]:+.0f} mm above {L}" if now is not None else ""))
                if now is None:
                    now = vision.measure_tip(5, display.show)
                start = now["pos"][[0, 2]] if now is not None else tip_rest["pos"][[0, 2]]
                display.lines = [status_line, f"{L}: moving over it"]
                waypoints, path, avoided = plan_route(vision, start, target, L, obstacles, reachable, args)
                display.path = (path, target[1])
                print(f"  route: {'straight' if len(path) == 2 else f'detour via {len(path) - 2} corner(s)'}, "
                      f"{path_length(path):.0f} mm, {len(waypoints)} waypoint(s), "
                      f"contact with {len(touched(path, [o for o in avoided if o['kind'] == 'object'], args.hug_mm + 3))} object(s), "
                      f"closest approach to a letter {min_clearance(path, [o for o in avoided if o['kind'] != 'object']):.0f} mm")
                # 2. move over it (the model keeps learning from every move; staying high is only a
                #    weak extra goal, the direction to the letter comes first)
                outcome = move_tip_along(cables, vision, waypoints, target, display.show, tol_mm=args.success_mm,
                                         adapt=True, hold_y=hold_y)
                # 3. come down onto it if it was raised
                if hold_y is not None and outcome["reason"] != "lost the tip":
                    display.lines = [status_line, f"{L}: lowering onto it"]
                    print(f"  over {L}: {outcome['error_mm']:.0f} mm off sideways; lowering onto it")
                    final = lower_onto(cables, vision, target, display.show, tol_mm=args.success_mm)
                    print(f"  lowered the tip {final['lowered_mm']:.0f} mm ({final['settle']})")
                    outcome = {**final, "steps": outcome["steps"] + final["steps"], "waypoints": outcome["waypoints"],
                               "reason": "reached" if final["reached"] else final["reason"]}
                # 4. stopped short: the model may be off this far from where it was measured, so
                #    measure it again here and try once more
                if not outcome["reached"] and outcome["reason"] in ("out of reach", "step limit"):
                    J, _ = probe_model(cables, vision, display.show)
                    if J is not None:
                        set_live_model(cables, J)
                        print(f"  stopped {outcome['error_mm']:.0f} mm short; cables here: " + describe_model(J, cables.used > 0))
                        again = move_tip_adaptive(cables, vision, target, display.show, tol_mm=args.success_mm, max_iter=15)
                        outcome = {**again, "steps": outcome["steps"] + again["steps"], "waypoints": outcome["waypoints"]}
                seconds = time.time() - t0
                display.path = None
                status[L] = "OK" if outcome["reached"] else "X"
                tip = outcome["tip"] if outcome["tip"] is not None else [np.nan] * 3
                rows.append({**row, "success": int(outcome["reached"]), "error_mm": round(outcome["error_mm"], 1),
                             "seconds": round(seconds, 2), "steps": outcome["steps"], "reason": outcome["reason"],
                             "target_x": round(target[0], 1), "target_y": round(target[1], 1),
                             "target_z": round(target[2], 1), "tip_x": round(tip[0], 1),
                             "tip_y": round(tip[1], 1), "tip_z": round(tip[2], 1),
                             "waypoints": outcome["waypoints"], "path_mm": round(path_length(path), 1),
                             "min_clearance_mm": round(min_clearance(path, [o for o in avoided if o["kind"] != "object"]), 1)
                             if any(o["kind"] != "object" for o in avoided) else "",
                             "raised_mm": round(raised["raised_mm"], 1),
                             "above_letter_mm": round(tip[1] - target[1], 1)})
                print(f"[trial {trial}] {L}: {'SUCCESS' if outcome['reached'] else 'FAILED'} in {seconds:.1f} s, "
                      f"final error {outcome['error_mm']:.1f} mm ({outcome['reason']}), "
                      f"tip {tip[1] - target[1]:+.0f} mm above the letter")

                display.lines = ["   ".join(f"{k} {v}" for k, v in status.items()) + f"   (trial {trial})"]
                for _ in range(int(args.hold * 30)):          # pause over the letter (~30 fps)
                    frame, depth = vision.grab()
                    display.show(frame, vision.find_tip(frame, depth))
                if not outcome["reached"]:
                    # it may be caught on something with the cables under tension: start the next
                    # letter from the rest pose, not from there
                    cables.set_pulls(np.zeros(4))

            display.target_px = display.target_pos = None
            cables.set_pulls(np.zeros(4))                     # back to the rest pose
            total = time.time() - trial_start
            display.timer_start = None
            reached = sum(r["success"] for r in rows if r["trial"] == trial and r["letter"] in SEQUENCE)
            rows.append({"time": datetime.now().isoformat(timespec="seconds"), "trial": trial, "letter": "ALL",
                         "success": int(reached == len(SEQUENCE)), "seconds": round(total, 2),
                         "reason": f"{reached}/{len(SEQUENCE)} letters", "success_mm": args.success_mm})
            print(f"\n[trial {trial}] {reached}/{len(SEQUENCE)} letters reached in {total:.1f} s")
    except KeyboardInterrupt:
        print("Stopped.")
    except (RuntimeError, ValueError) as e:
        print(f"Stopped: {e}")
    finally:
        cables.close()
        camera.close()
        if recorder is not None:
            recorder.close()
        cv2.destroyAllWindows()
        if rows:
            save_and_summarise(rows)


# ---------------------------------------------------------------- jog

def cmd_jog(cfg, args):
    print("Wind each cable by keyboard to take up slack or to find which direction pulls.\n"
          "In the camera window: 0-3 select a motor | w / s = +1 / -1 mm | W / S = +5 / -5 mm | q quit.\n"
          "A cable is just taut when the tip starts to move. Note for each motor whether + or - tightened it.")
    camera = open_camera_or_exit()
    cables = Cables(max_travel_rad=5.0)              # by hand: allow up to ~100 mm per cable
    vision = Vision(cfg, camera)
    display = Display(vision)
    steps_mm = {ord("w"): 1.0, ord("s"): -1.0, ord("W"): 5.0, ord("S"): -5.0}
    try:
        if not cables.open():
            print("Could not connect to the motors (power on? USB plugged in? EMIO Labs closed?).")
            return
        start = cables.read_angles()
        target = start.copy()
        motor = 0
        base = vision.measure_tip(5)
        while True:
            frame, depth = vision.grab()
            tip = vision.find_tip(frame, depth)
            change = (target - start) * PULLEY_RADIUS_MM
            moved = (f"{np.linalg.norm(tip['px'] - base['px']):.0f} px"
                     if tip is not None and base is not None else "? (tip not visible)")
            display.lines = [f"motor {motor} selected  (0-3 select, w/s +-1 mm, W/S +-5 mm, q quit)",
                             "cable turned: " + "  ".join(f"M{j} {change[j]:+.0f} mm" for j in range(4)),
                             f"tip moved {moved} since start"]
            key = display.show(frame, tip)
            if ord("0") <= key <= ord("3"):
                motor = key - ord("0")
            elif key in steps_mm:
                target[motor] += steps_mm[key] / PULLEY_RADIUS_MM
                cables.move_angles(target, timeout=1.5)
                actual = cables.read_angles()[motor]
                print(f"M{motor}: {(target[motor] - start[motor]) * PULLEY_RADIUS_MM:+.0f} mm commanded, "
                      f"{(actual - start[motor]) * PULLEY_RADIUS_MM:+.0f} mm actual")
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError) as e:
        print(f"Stopped: {e}")
    finally:
        cables.close()            # torque off; the gearboxes keep the cables where they are
        camera.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------- goto

def cmd_goto(cfg, args):
    calib = load_calibration()
    if calib is None:
        sys.exit("No cable_calibration.json yet: run `python nus_main.py calibrate` first.")
    print(f"Using calibration from {calib['created']} (recalibrate if the cables were re-wound).")

    camera = open_camera_or_exit()
    vision = Vision(cfg, camera)
    display = Display(vision)
    cables = None
    try:
        found = locate_letters(vision, display)
        order = list(LETTERS) if args.letter == "ALL" else [args.letter]
        J = np.array(calib["J"])[[0, 2], :]
        tip_taut = np.array(calib["tip_taut"])
        for L in order:
            if L not in found:
                print(f"{L}: not found")
                continue
            pos = found[L]["pos"]
            p, residual = solve_pulls(J, pos[[0, 2]] - tip_taut[[0, 2]], calib["pull_max_mm"])
            print(f"{L}: at {fmt(pos)} (score {found[L]['score']:.2f}) | "
                  f"{np.hypot(*(pos[[0, 2]] - tip_taut[[0, 2]])):.0f} mm sideways from the hanging tip | "
                  f"planned pulls {np.round(p, 1)} mm"
                  + (f" | WARNING: model says {residual:.0f} mm short (likely out of reach)" if residual > 5 else ""))
        if args.dry_run:
            print("Dry run: no motor motion.")
            return

        cables = Cables(calib)
        if not cables.open():
            print("Could not connect to the motors (power on? USB plugged in? EMIO Labs closed?).")
            return
        align_to_calibration(cables)            # refuses if a spool moved since calibration
        cables.set_pulls(np.zeros(4))
        for L in order:
            if L not in found:
                continue
            target = found[L]["pos"]
            display.target_px = found[L]["px"]
            display.lines = [f"target {L} at {fmt(target)}"]
            print(f"\nMoving over {L}...")
            outcome = move_tip_to(cables, vision, target, display.show, tol_mm=args.tol)
            print(f"{L}: {'reached' if outcome['reached'] else 'NOT reached'} ({outcome['reason']}, "
                  f"error {outcome['error_mm']:.1f} mm)")
            for _ in range(int(args.hold * 30)):          # hold over the letter (~30 fps)
                frame, depth = vision.grab()
                display.show(frame, vision.find_tip(frame, depth))
            cables.set_pulls(np.zeros(4))
    except KeyboardInterrupt:
        print("Stopped.")
    except (RuntimeError, ValueError) as e:
        print(f"Stopped: {e}")
    finally:
        if cables is not None:
            cables.close()
        camera.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="N-U-S letter retrieval with the EMIO cable manipulator")
    sub = parser.add_subparsers(dest="command", required=True)
    p_chk = sub.add_parser("check", help="camera only: check a (new) set, measure the surface height")
    p_chk.add_argument("--apply", action="store_true", help="save the measured stage height to nus_config.json")
    p_chk.add_argument("--letter-mm", type=float, default=5.0, help="letter thickness in mm (default 5)")
    sub.add_parser("detect", help="camera only: show detections, teach letter shapes")
    sub.add_parser("jog", help="wind cables by keyboard: take up slack, find pull directions")
    p_cal = sub.add_parser("calibrate", help="find cable directions/taut points and measure the model")
    p_cal.add_argument("--first-direction", type=int, choices=(-1, 1), default=-1,
                       help="direction to try first when searching for each cable's pull direction")
    p_cal.add_argument("--pull-signs", type=parse_pull_signs, default=None,
                       help="known pull direction per motor, e.g. --pull-signs=-1,-1,1,1 (only that "
                            "direction is searched, so no cable is unwound)")
    p_cal.add_argument("--search-mm", type=float, default=30.0,
                       help="how much cable to turn while looking for the taut point (default 30, max 100)")
    p_go = sub.add_parser("goto", help="steer the tip over a letter")
    p_go.add_argument("letter", type=str.upper, choices=(*LETTERS, "ALL"))
    p_go.add_argument("--dry-run", action="store_true", help="detect and plan only, never move")
    p_go.add_argument("--tol", type=float, default=4.0, help="position tolerance in mm (default 4)")
    p_go.add_argument("--hold", type=float, default=2.0, help="seconds to hold over each letter")
    p_s = sub.add_parser("search", help="N -> U -> S: move the tip to each letter in turn, timed and logged")
    p_s.add_argument("--success-mm", type=float, default=8.0,
                     help="tip within this horizontal distance of the letter counts as reached (default 8)")
    p_s.add_argument("--trials", type=int, default=1, help="how many N-U-S runs (letters can be moved between)")
    p_s.add_argument("--hold", type=float, default=1.0, help="seconds to pause over each reached letter")
    p_s.add_argument("--record", action="store_true", help="save the annotated camera view as an MP4 in runs/")
    p_s.add_argument("--route", choices=("contact", "avoid", "straight"), default="contact",
                     help="contact (default): lean on rigid objects so the arm bends round them, keep clear of letters; avoid: keep clear of everything; straight: no planning")
    p_s.add_argument("--hug-mm", type=float, default=8.0,
                     help="contact route: tip path offset from an object's edge, about the arm's radius (default 8)")
    p_s.add_argument("--clearance-mm", type=float, default=15.0,
                     help="how far the tip keeps from other letters (and objects with --route avoid; default 15)")
    p_s.add_argument("--auto-start", type=float, default=None, metavar="SECONDS",
                     help="start by itself once all letters have been found for this long (no SPACE needed)")
    p_s.add_argument("--waypoint-step-mm", type=float, default=15.0,
                     help="spacing of the points the tip follows along its route (default 15)")
    p_s.add_argument("--no-raise", dest="raise_tip", action="store_false",
                     help="don't bend the tip up before moving (go straight over at its normal height)")
    p_s.add_argument("--raise-above-mm", type=float, default=20.0,
                     help="raise the tip until it is this far above the letter it goes to (it goes around obstacles; default 20; "
                          "0 = as high as it goes)")
    p_s.add_argument("--raise-max-ma", type=float, default=300.0,
                     help="stop raising when a motor draws this much more current (mA, default 300): cables getting tight")
    p_s.add_argument("--pull-max-mm", type=float, default=None,
                     help=f"most each cable may pull, in mm (default: the calibration's, {PULL_MAX_MM:.0f}); "
                          "more lets the arm bend further (reach further, raise the tip higher)")
    args = parser.parse_args()
    if args.command == "calibrate" and not 1 <= args.search_mm <= 100:
        parser.error("--search-mm must be between 1 and 100")
    if args.command == "search":
        limit = MAX_TRAVEL_RAD * PULLEY_RADIUS_MM - 10       # margin for spool drift since calibration
        if args.pull_max_mm is not None and not 1 <= args.pull_max_mm <= limit:
            parser.error(f"--pull-max-mm must be between 1 and {limit:.0f} mm of cable")

    cfg = load_config()
    commands = {"check": cmd_check, "detect": cmd_detect, "jog": cmd_jog, "calibrate": cmd_calibrate,
                "goto": cmd_goto, "search": cmd_search}
    commands[args.command](cfg, args)


def parse_pull_signs(text):
    try:
        signs = [int(s) for s in text.replace(" ", "").split(",")]
    except ValueError:
        signs = []
    if len(signs) != 4 or any(s not in (-1, 0, 1) for s in signs):
        raise argparse.ArgumentTypeError("give four values of -1, 1 (or 0 = motor unused), "
                                         "e.g. --pull-signs=0,-1,1,1")
    return signs


if __name__ == "__main__":
    main()
