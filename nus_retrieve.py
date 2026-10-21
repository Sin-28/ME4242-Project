"""
Retrieve the N, U, S letters into their holder, by trial and error with path planning.

    python nus_main.py holder               # camera only, once per set-up: click each holder slot, press n / u / s
    python nus_main.py retrieve --dry-run   # camera only: letters, obstacles and the planned routes
    python nus_main.py retrieve             # pick the letters up one by one and put them in the holder
    python nus_main.py retrieve --no-place  # only reach each magnet (no holder yet)

Until all three letters are in their slots (or Ctrl+C / q in the camera window):
 1. Scan: the letters (pick point = magnet) and the obstacles, over a few frames.
 2. Plan: for every letter still to do, one route per approach side of the block it lies on (or
    of the letter): the shortest path around the obstacles to just outside that side, then
    straight in to the magnet. The obstacles are grown by the tip's length too (the magnet tip
    sticks out past the green marker that the camera tracks). Sides facing away from the arm's
    middle are preferred: a bent arm's tip is higher the further out it reaches, so it gets onto
    the block more easily from there. The cheapest route over all letters is tried first.
 3. Try it. A watchdog gives up when the tip hasn't moved for --stuck-s seconds (it's caught).
 4. Reached: touch the magnet, check the letter came along (it's gone from where it lay), take it
    to its slot the same way, put it down, back off to the middle, check it's in the slot.
 5. Not reached / stuck / lost the tip / grab or placing failed: cables released (the tip goes
    back to the middle), that side is marked as failed, rescan, try the next best route.
"""

import csv
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from nus_control import (Cables, Watchdog, align_to_calibration, describe_model, load_calibration, lower_onto,
                         move_tip_adaptive, move_tip_along, probe_model, reachability, reset_live_model,
                         set_live_model)
from nus_path import approach_routes, densify, footprint, support_of
from nus_ui import WINDOW, Display, Recorder, open_camera_or_exit, stable_objects
from nus_vision import LETTERS, Vision, fmt, save_config_value

RUNS_DIR = Path(__file__).resolve().parent / "runs"
LOG_FIELDS = ["time", "letter", "attempt", "stage", "side_deg", "cost", "result", "reason", "seconds",
              "target_x", "target_y", "target_z", "tip_x", "tip_y", "tip_z"]
COLOURS = {"N": (60, 60, 230), "U": (40, 40, 40), "S": (230, 120, 30)}


# ---------------------------------------------------------------- scene

def scan_scene(vision, display, todo, frames=12, wait_s=8.0, lines=()):
    """Letters (median over the last `frames` frames, pick point = magnet) and stable obstacles.
    Keeps looking for up to wait_s seconds until every letter in `todo` has been seen."""
    seen = {L: deque(maxlen=frames) for L in LETTERS}
    objects_seen = deque(maxlen=5)
    start, n, letter_obstacles = time.time(), 0, {}
    while True:
        frame, depth = vision.grab()
        detections, candidates = vision.detect_letters(frame, depth)
        tip = vision.find_tip(frame, depth)
        obstacles = vision.detect_obstacles(frame, depth, list(detections.values()), tip=tip)
        objects_seen.append([o for o in obstacles if o["kind"] == "object"])
        letter_obstacles.update({o["kind"]: o for o in obstacles if o["kind"] in LETTERS})
        for L in LETTERS:
            d = detections.get(L)
            seen[L].append(d if d is not None and d["pos"] is not None else None)
        n += 1
        display.candidates, display.detections = candidates, detections
        display.obstacles = stable_objects(objects_seen) + list(letter_obstacles.values())
        display.lines = list(lines) + ["scanning the letters and obstacles..."]
        display.show(frame, tip)
        found = {L: [d for d in seen[L] if d is not None] for L in LETTERS}
        if n >= frames and (all(len(found[L]) >= frames // 2 for L in todo) or time.time() - start > wait_s):
            break
    letters = {L: {"pos": np.median([d["pos"] for d in ds], axis=0), "px": np.median([d["px"] for d in ds], axis=0)}
               for L, ds in found.items() if len(ds) >= max(2, frames // 3)}
    objects = stable_objects(objects_seen)
    for L, f in letters.items():                      # letter footprint (a circle) for the planner
        o = letter_obstacles.get(L) or {"centre": f["pos"][[0, 2]], "radius": 25.0, "height_y": f["pos"][1]}
        f["obstacle"] = dict(o, kind=L)
    return letters, objects


def in_slot(letter, slots, L, tol_mm):
    return L in slots and np.hypot(*(np.asarray(letter["pos"])[[0, 2]] - np.asarray(slots[L])[[0, 2]])) < tol_mm


# ---------------------------------------------------------------- planning

def plan(ctx, start_xz, goal_xyz, letters, objects, skip=(), failed=(), carrying=False):
    """Candidate routes from start_xz to goal_xyz (see nus_path.approach_routes). Obstacles: every
    object except the one under the goal (the support, crossed only on the last leg), grown by the
    tip's length too, and the letters not in `skip` (+ the carried letter's half-size when carrying)."""
    args, cfg = ctx["args"], ctx["cfg"]
    goal = np.asarray(goal_xyz, float)[[0, 2]]
    extra = cfg["tip_length_mm"] + (args.carry_margin_mm if carrying else 0.0)
    support = support_of(goal, objects)
    hug = args.hug_mm if args.route == "contact" else args.clearance_mm
    others = [dict(o, inflate=hug + extra) for o in objects if o is not support]
    # letters: the clearance only (the tip's length matters at rigid edges, where it catches)
    others += [dict(f["obstacle"], inflate=args.clearance_mm + (args.carry_margin_mm if carrying else 0.0))
               for L, f in letters.items() if L not in skip]
    if support is not None:
        support = dict(support, inflate=cfg["tip_length_mm"] + 2.0)
    return approach_routes(start_xz, goal, others, support=support, axis=ctx["middle"],
                           clearance_mm=args.clearance_mm, standoff_mm=cfg["tip_length_mm"] + 6.0,
                           directions=args.directions, reachable=ctx["reachable"], failed_angles=failed), support


def show_plans(display, plans, chosen=None, marks=()):
    """Chosen route in cyan, each letter's best route (and the chosen letter's alternatives) in grey,
    approach points as dots (plus the given marks, e.g. the holder's slots)."""
    display.alt_paths, display.marks = [], list(marks)
    for L, (routes, y) in plans.items():
        for k, r in enumerate(routes[:4] if L == chosen else routes[:1]):
            if L == chosen and k == 0:
                display.path = (r["path"], y)
            else:
                display.alt_paths.append((r["path"], y))
            display.marks.append(((r["approach"][0], y, r["approach"][1]), f"{L}{r['angle_deg']:.0f}", COLOURS[L]))


# ---------------------------------------------------------------- moving

def back_to_middle(ctx, settle_s=1.2):
    """Release the cables: the arm goes back to its middle (rest) pose. Model back to the calibration's."""
    ctx["cables"].set_pulls(np.zeros(4))
    end = time.time() + settle_s
    while time.time() < end:
        frame, depth = ctx["vision"].grab()
        ctx["display"].show(frame, ctx["vision"].find_tip(frame, depth))
    reset_live_model(ctx["cables"])


def drive(ctx, route, target_xyz, tol_mm):
    """Follow a planned route to target_xyz, giving up if the tip gets caught (Watchdog)."""
    args = ctx["args"]
    watch = Watchdog(args.stuck_s, args.stuck_mm, args.attempt_timeout_s)
    waypoints = densify(route["path"], args.waypoint_step_mm)
    out = move_tip_along(ctx["cables"], ctx["vision"], waypoints, np.asarray(target_xyz, float), ctx["display"].show,
                         tol_mm=tol_mm, adapt=True, watch=watch, log=lambda *a, **k: None)
    if out["reason"] == "stuck":
        out["reason"] = f"stuck: {watch.why}"
    return out


def measure_here(ctx):
    """Measure the cable model where the arm is now (it is bent: the calibration's model is for the middle)."""
    J, _ = probe_model(ctx["cables"], ctx["vision"], ctx["display"].show)
    if J is not None:
        set_live_model(ctx["cables"], J)
        print("    cables here: " + describe_model(J, ctx["cables"].used > 0))


def touch(ctx, target_xyz):
    """Line up closely over the magnet, come down onto it (the marker stops tip_length above it)
    and hold a moment, so the tip's magnet takes the letter."""
    args = ctx["args"]
    move_tip_adaptive(ctx["cables"], ctx["vision"], np.asarray(target_xyz, float), ctx["display"].show,
                      tol_mm=min(4.0, args.success_mm), max_iter=6, log=lambda *a, **k: None,
                      watch=Watchdog(args.stuck_s, args.stuck_mm, 15.0))
    goal = np.asarray(target_xyz, float) + [0.0, ctx["cfg"]["tip_length_mm"], 0.0]
    out = lower_onto(ctx["cables"], ctx["vision"], goal, ctx["display"].show, tol_mm=ctx["args"].success_mm,
                     log=lambda *a, **k: None)
    hold(ctx, 0.6)
    return out


def hold(ctx, seconds):
    end = time.time() + seconds
    while time.time() < end:
        frame, depth = ctx["vision"].grab()
        ctx["display"].show(frame, ctx["vision"].find_tip(frame, depth))


def where_is(ctx, L, frames=8):
    """Median position of letter L over a few frames, or None if it isn't seen lying anywhere."""
    found = []
    for _ in range(frames):
        frame, depth = ctx["vision"].grab()
        detections, candidates = ctx["vision"].detect_letters(frame, depth)
        ctx["display"].candidates, ctx["display"].detections = candidates, detections
        ctx["display"].show(frame, ctx["vision"].find_tip(frame, depth))
        if L in detections and detections[L]["pos"] is not None:
            found.append(detections[L]["pos"])
    return np.median(found, axis=0) if len(found) >= frames // 3 else None


def grabbed(ctx, L, lay_at):
    """Back off a little toward the middle and look: a letter still lying where it was wasn't picked up."""
    tip = ctx["vision"].measure_tip(3, ctx["display"].show)
    if tip is not None:
        here = tip["pos"][[0, 2]]
        away = here + 25.0 * (ctx["middle"] - here) / max(np.linalg.norm(ctx["middle"] - here), 1e-6)
        move_tip_adaptive(ctx["cables"], ctx["vision"], np.array([away[0], tip["pos"][1], away[1]]),
                          ctx["display"].show, tol_mm=6.0, max_iter=8, log=lambda *a, **k: None,
                          watch=Watchdog(ctx["args"].stuck_s, ctx["args"].stuck_mm, 20.0))
    now = where_is(ctx, L)
    return now is None or np.hypot(*(now[[0, 2]] - lay_at[[0, 2]])) > 12.0


# ---------------------------------------------------------------- the loop

def cmd_retrieve(cfg, args):
    calib = load_calibration()
    if calib is None:
        sys.exit("No cable_calibration.json yet: run `python nus_main.py calibrate` first.")
    slots = {L: np.asarray(p, float) for L, p in (cfg.get("holder_slots") or {}).items()}
    if not args.no_place and not args.dry_run and set(slots) != set(LETTERS):
        sys.exit("The holder's slots aren't taught yet (" + ", ".join(sorted(set(LETTERS) - set(slots))) +
                 " missing): run `python nus_main.py holder`, or `retrieve --no-place` to only reach the magnets.")
    if args.tip_mm is not None:
        cfg["tip_length_mm"] = args.tip_mm
    if args.pull_max_mm is not None:
        calib = {**calib, "pull_max_mm": [args.pull_max_mm if s else 0.0 for s in calib["pull_sign"]]}

    camera = open_camera_or_exit()
    vision = Vision(cfg, camera)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    recorder = Recorder(RUNS_DIR / f"retrieve_{stamp}.mp4") if args.record else None
    display = Display(vision, recorder)
    cables = None if args.dry_run else Cables(calib)
    rows = []
    try:
        if cables is not None:
            if not cables.open():
                print("Could not connect to the motors (power on? USB plugged in? EMIO Labs closed?).")
                return
            align_to_calibration(cables)            # refuses if a spool moved since calibration
            cables.calib = {**cables.calib, "pull_max_mm": calib["pull_max_mm"]}
            cables.set_pulls(np.zeros(4))
            reset_live_model(cables)
        rest = vision.measure_tip(10, display.show)
        if rest is None:
            print("Can't see the green tip. Check the camera view.")
            return
        ctx = {"cfg": cfg, "args": args, "vision": vision, "display": display, "cables": cables,
               "middle": rest["pos"][[0, 2]].copy(),
               # approach points are loose waypoints: within ~25 mm of what the cables reach is good enough
               "reachable": reachability(calib, rest["pos"], tol_mm=25.0)}
        ctx["slot_marks"] = [(tuple(p), f"slot {L}", COLOURS[L]) for L, p in slots.items()]
        display.marks = list(ctx["slot_marks"])
        print(f"Tip at the middle: {fmt(rest['pos'])}. Tip length past the marker: {cfg['tip_length_mm']:.0f} mm. "
              + ("Holder slots: " + ", ".join(f"{L} {fmt(p)}" for L, p in slots.items()) if slots else "No holder slots."))

        placed, failed, attempts = set(), {L: [] for L in LETTERS}, {L: 0 for L in LETTERS}
        second_round = False
        t0 = time.time()
        while True:
            todo = [L for L in LETTERS if L not in placed]
            if not todo:
                print(f"\nAll three letters done in {time.time() - t0:.0f} s.")
                break
            status = "   ".join(f"{L} {'done' if L in placed else f'try {attempts[L]}'}" for L in LETTERS)
            letters, objects = scan_scene(vision, display, todo, lines=[status])
            for L in todo:
                if slots and L in letters and in_slot(letters[L], slots, L, args.slot_tol_mm):
                    print(f"{L} is already in its slot.")
                    placed.add(L)
            todo = [L for L in todo if L not in placed]
            if not todo:
                continue

            # plan every letter still to do, from the middle
            plans = {}
            for L in todo:
                if L not in letters:
                    print(f"{L}: not found in the camera view")
                    continue
                if attempts[L] >= args.max_attempts:
                    continue
                routes, _ = plan(ctx, ctx["middle"], letters[L]["pos"], letters, objects, skip={L} | placed,
                                 failed=failed[L])
                if routes:
                    plans[L] = (routes, letters[L]["pos"][1])
            if not plans:
                if not second_round and any(failed[L] for L in todo):
                    print("Every route tried has failed: trying them all once more (the letters may have moved).")
                    second_round = True
                    failed = {L: [] for L in LETTERS}
                    attempts = {L: 0 for L in LETTERS}
                    continue
                print("No route left to try for " + ", ".join(todo) + ". Stopping.")
                break
            L = min(plans, key=lambda k: plans[k][0][0]["cost"])
            route = plans[L][0][0]
            show_plans(display, plans, chosen=L, marks=ctx["slot_marks"])
            print(f"\nPlan: " + " | ".join(f"{k} best side {r[0]['angle_deg']:.0f} deg, {r[0]['length']:.0f} mm "
                                          f"(cost {r[0]['cost']:.0f}, {len(r)} side(s) left)" for k, (r, _) in plans.items()))

            if args.dry_run:
                display.lines = [f"DRY RUN: would go to {L} first (side {route['angle_deg']:.0f} deg)  q = quit",
                                 "cyan = chosen route, grey = other candidates, dots = approach points"]
                end = time.time() + args.dry_run_s
                while time.time() < end:
                    frame, depth = vision.grab()
                    display.show(frame, vision.find_tip(frame, depth))
                break

            attempts[L] += 1
            target = letters[L]["pos"]
            display.target_px, display.target_pos = letters[L]["px"], target
            display.lines = [status, f"{L}: attempt {attempts[L]}, from side {route['angle_deg']:.0f} deg"]
            print(f"{L}: attempt {attempts[L]} from side {route['angle_deg']:.0f} deg, "
                  f"{len(route['path']) - 2} corner(s), {route['length']:.0f} mm, magnet at {fmt(target)}")
            started = time.time()
            out = drive(ctx, route, target, args.success_mm)
            stage, ok = "reach", out["reached"]
            if ok:
                touch(ctx, target)
                if args.no_place:
                    print(f"  {L}: reached its magnet ({out['error_mm']:.1f} mm off)")
                    placed.add(L)
                else:
                    stage = "grab"
                    ok = grabbed(ctx, L, target)
                    out["reason"] = "picked up" if ok else "letter didn't come along"
                    if ok:
                        stage = "place"
                        ok, out = place(ctx, L, slots[L], letters, objects, placed)
                        if ok:
                            placed.add(L)
            rows.append(log_row(L, attempts[L], stage, route, ok, out, time.time() - started, target))
            print(f"  {L}: {'OK' if ok else 'FAILED'} at {stage} ({out['reason']})")
            if not ok:
                failed[L].append(route["angle_deg"])
            display.path = display.target_px = display.target_pos = None
            back_to_middle(ctx)                      # every attempt starts from the middle
    except KeyboardInterrupt:
        print("Stopped.")
    except (RuntimeError, ValueError) as e:
        print(f"Stopped: {e}")
    finally:
        if cables is not None:
            cables.close()
        camera.close()
        if recorder is not None:
            recorder.close()
        cv2.destroyAllWindows()
        if rows:
            save_log(rows)


def place(ctx, L, slot, letters, objects, placed):
    """Carry the picked-up letter to its slot (planned like a letter: the holder is the support),
    put it down, back off to the middle, and check it is there. Returns (ok, outcome)."""
    args = ctx["args"]
    measure_here(ctx)
    tip = ctx["vision"].measure_tip(3, ctx["display"].show)
    start = tip["pos"][[0, 2]] if tip is not None else ctx["middle"]
    routes, _ = plan(ctx, start, slot, letters, objects, skip={L} | placed, carrying=True)
    out = {"reached": False, "reason": "no route to the slot", "error_mm": np.nan, "tip": None}
    for route in routes[:args.place_tries]:
        ctx["display"].path, ctx["display"].target_pos = (route["path"], slot[1]), slot
        print(f"    carrying {L} to its slot from side {route['angle_deg']:.0f} deg")
        out = drive(ctx, route, slot, args.success_mm)
        if out["reached"]:
            break
        back_to_middle(ctx)                          # the letter stays on the magnet
    if not out["reached"]:
        return False, out
    touch(ctx, slot)
    back_to_middle(ctx)                              # the slot keeps the letter, the magnet lets go
    now = where_is(ctx, L)
    if now is not None and np.hypot(*(now[[0, 2]] - slot[[0, 2]])) < args.slot_tol_mm:
        return True, {**out, "reason": "in its slot"}
    if now is not None:
        letters[L]["pos"] = now
        return False, {**out, "reason": f"dropped at {fmt(now)}, not in the slot"}
    return False, {**out, "reason": "not seen after letting go (still on the magnet?)"}


def log_row(L, attempt, stage, route, ok, out, seconds, target):
    tip = out.get("tip") if out.get("tip") is not None else [np.nan] * 3
    return {"time": datetime.now().isoformat(timespec="seconds"), "letter": L, "attempt": attempt, "stage": stage,
            "side_deg": round(route["angle_deg"]), "cost": round(route["cost"], 1), "result": int(ok),
            "reason": out.get("reason"), "seconds": round(seconds, 1),
            "target_x": round(target[0], 1), "target_y": round(target[1], 1), "target_z": round(target[2], 1),
            "tip_x": round(tip[0], 1), "tip_y": round(tip[1], 1), "tip_z": round(tip[2], 1)}


def save_log(rows):
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / "retrieve_log.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
    done = sorted({r["letter"] for r in rows if r["result"] and r["stage"] in ("place", "reach")})
    print(f"\n{len(rows)} attempt(s), done: {', '.join(done) or 'none'}. Log appended to {path}")


# ---------------------------------------------------------------- teach the holder

def cmd_holder(cfg, args):
    """Click on each slot of the holder in the camera window and press n / u / s: its position
    (from the depth camera) is saved as holder_slots in nus_config.json."""
    camera = open_camera_or_exit()
    vision = Vision(cfg, camera)
    display = Display(vision)
    slots = {L: list(p) for L, p in (cfg.get("holder_slots") or {}).items()}
    state = {"click": None, "depth": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.setMouseCallback(WINDOW, on_mouse)
    print("Click the middle of a slot (where that letter's magnet ends up), then press n / u / s. "
          "c = clear all, q = save and quit.")
    try:
        while True:
            frame, depth = vision.grab()
            state["depth"] = depth
            display.marks = [(tuple(p), f"slot {L}", COLOURS[L]) for L, p in slots.items()]
            display.lines = ["holder slots: " + (", ".join(f"{L} {fmt(p)}" for L, p in slots.items()) or "none"),
                             "click a slot, press n / u / s   c = clear   q = save and quit"]
            if state["click"] is not None:
                cv2.circle(frame, state["click"], 4, (255, 0, 255), -1)
            key = display.show(frame, vision.find_tip(frame, depth))
            if key in (ord("n"), ord("u"), ord("s")) and state["click"] is not None:
                L = chr(key).upper()
                u, v = state["click"]
                window = depth[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4]
                d = window[window > 0]
                pos = (vision.pixel_to_robot(u, v, float(np.median(d))) if len(d) >= 5
                       else vision.pixel_to_plane(u, v, cfg["stage_y_mm"]))
                if pos is None:
                    print("Couldn't get a position there: click on the holder.")
                    continue
                slots[L] = [round(float(c), 1) for c in pos]
                print(f"Slot {L}: {fmt(pos)}")
            elif key == ord("c"):
                slots = {}
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()
    save_config_value("holder_slots", slots)
    print(f"Saved holder_slots ({', '.join(sorted(slots)) or 'none'}) to nus_config.json")
