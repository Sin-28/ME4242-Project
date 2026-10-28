"""
Retrieve the N, U, S letters into their holder, by trial and error with path planning.

    python main.py holder               # camera only, once per set-up: click each holder slot, press n / u / s
    python main.py retrieve --dry-run   # camera only: letters, obstacles and the planned routes
    python main.py retrieve             # pick the letters up one by one and put them in the holder
    python main.py retrieve --no-place  # only reach each magnet (no holder yet)

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

from cable_control import (Cables, Watchdog, _cable_load, align_to_calibration, describe_model, load_calibration, lower_onto,
                         move_tip_adaptive, move_tip_along, probe_model, reachability, reset_live_model,
                         set_live_model)
from workspace_map import follow, load_map, play
from path_planning import (approach_routes, convex_hull, densify, footprint, path_length, plan_path, polygons_overlap,
                      relative_yaw, support_of)
from display import WINDOW, Display, Recorder, open_camera_or_exit, stable_objects
from vision import LETTERS, Vision, fmt, save_config_value

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
    start, n, letter_obstacles, last = time.time(), 0, {}, {}
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
            if d is not None and d["pos"] is not None:
                last[L] = d
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
        f["outline"] = vision.letter_outline(last[L]) if L in last else None   # x-z, relative to its magnet
    return letters, objects


def orientation(ctx, L, letter):
    """(turn in deg, overlap) to bring the letter as it lies into the orientation of its slot, or None."""
    ref = ctx.get("slot_outlines", {}).get(L)
    if ref is None or letter.get("outline") is None:
        return None
    return relative_yaw(letter["outline"], ref)


def in_slot(letter, slots, L, tol_mm):
    return L in slots and np.hypot(*(np.asarray(letter["pos"])[[0, 2]] - np.asarray(slots[L])[[0, 2]])) < tol_mm


# ---------------------------------------------------------------- planning

def plan(ctx, start_xz, goal_xyz, letters, objects, skip=(), failed=(), carrying=False, start_pulls=None,
         carried=None):
    """Candidate routes from start_xz (cables at start_pulls, default all let out) to goal_xyz.
    With a workspace map: the cheapest chain of recorded pull settings to the goal (the obstacles
    were in place while exploring, so contacts are in it), keeping clear of the letters not in
    `skip`; places where earlier tries failed (`failed`: {"angle", "spot", "approach"}) are
    avoided if possible. Without one: one route per approach side (path_planning.approach_routes),
    around every object except the one under the goal, grown by the tip's length, and the letters."""
    args, cfg = ctx["args"], ctx["cfg"]
    goal = np.asarray(goal_xyz, float)[[0, 2]]
    touch_y = float(goal_xyz[1]) + cfg["tip_length_mm"]           # marker height when the magnet touches
    # the carried letter hangs on the tip: its footprint (outline as it was picked up; the arm can't
    # turn it) must clear the obstacles, unless it is carried higher than them
    outline = None if carried is None else carried.get("outline")
    if carrying and outline is None:
        outline = circle_outline(25.0)                             # shape unknown: a 25 mm disk
    reach_out = 0.0 if outline is None else float(np.linalg.norm(outline, axis=1).max())
    margin = reach_out + args.carry_margin_mm if carrying else 0.0
    wmap = ctx.get("map")
    if wmap is not None and wmap.has_routes():
        blocked = [(np.asarray(f["obstacle"]["centre"], float), f["obstacle"]["radius"] + args.clearance_mm + margin)
                   for L, f in letters.items() if L not in skip]
        penalties = [(spot, 20.0, 300.0) for f in failed for spot in (f.get("spot"), f.get("approach"))
                     if spot is not None]
        lifted_from = None if carried is None or carried.get("pos") is None else np.asarray(carried["pos"])[[0, 2]]
        forbid, crossing = (carry_check(ctx, wmap, outline, objects, goal, lifted_from=lifted_from) if carrying
                            else (None, None))
        start_pulls = np.zeros(4) if start_pulls is None else start_pulls
        reach = args.success_mm + wmap.tolerance_at(goal)
        r = wmap.route(start_pulls, goal, blocked, penalties, forbid=forbid, goal_y=touch_y, crossing=crossing)
        if carrying and (r is None or r["end_gap"] > reach):
            # no way keeps the letter clear of everything: the way that brushes the fewest obstacles
            r = wmap.route(start_pulls, goal, blocked, penalties, forbid=forbid, goal_y=touch_y, crossing=crossing,
                           forbid_cost=200.0)
        if r is None or r["end_gap"] > reach:
            return [], None
        path = [np.asarray(p, float) for p in r["path"]] + [goal]
        before = path[-2]
        angle = float(np.degrees(np.arctan2(*(before - goal)[::-1]))) % 360.0
        return [{"path": path, "pulls": r["pulls"], "approach": before, "angle_deg": angle,
                 "cost": r["cost"], "length": r["length"], "graph": True, "end_height": r["end_height"],
                 "brushes": r["forbidden"]}], None
    failed = [f["angle"] for f in failed]
    extra = cfg["tip_length_mm"] + margin
    support = support_of(goal, objects)
    hug = args.hug_mm if args.route == "contact" else args.clearance_mm
    others = [dict(o, inflate=hug + extra) for o in objects if o is not support]
    # letters: the clearance only (the tip's length matters at rigid edges, where it catches)
    others += [dict(f["obstacle"], inflate=args.clearance_mm + margin)
               for L, f in letters.items() if L not in skip]
    if support is not None:
        support = dict(support, inflate=cfg["tip_length_mm"] + 2.0)
    routes = approach_routes(start_xz, goal, others, support=support, axis=ctx["middle"],
                             clearance_mm=args.clearance_mm, standoff_mm=cfg["tip_length_mm"] + 6.0,
                             directions=args.directions, reachable=ctx["reachable"], failed_angles=failed)
    if not routes:
        # no side works: as a last resort, the plain shortest way to the magnet (straight onto the
        # block, like `search`), tried once (its side is then marked failed like any other)
        start = np.asarray(start_xz, float)
        path = plan_path(start, goal, others, args.clearance_mm, ctx["reachable"]) or [start, goal]
        before = np.asarray(path[-2], float)
        angle = float(np.degrees(np.arctan2(*(before - goal)[::-1]))) % 360.0
        if not any(abs((angle - a + 180.0) % 360.0 - 180.0) < 30.0 for a in failed):
            length = path_length(path)
            routes = [{"path": list(path), "approach": before, "angle_deg": angle, "cost": length + 1000.0,
                       "length": length, "direct": True}]
    return routes, support


def circle_outline(radius, n=16):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def carry_check(ctx, wmap, outline, objects, goal, into_goal_mm=30.0, lifted_from=None):
    """Where the carried letter (convex outline + carry_margin_mm, at the tip) would hit an
    obstacle it isn't carried above. Returns (forbid: True per map setting, crossing(a, b):
    True if halfway between two settings it would). The last into_goal_mm before the goal are
    allowed (the letter goes into the holder there), and so is the block it was lifted from
    (it starts on top of that one and slides off it)."""
    args, cfg = ctx["args"], ctx["cfg"]
    shape = footprint({"polygon": convex_hull(outline)}, args.carry_margin_mm)
    below_marker = cfg["tip_length_mm"] + cfg["letter_thickness_mm"] + 3.0   # letter bottom under the marker
    under = None if lifted_from is None else support_of(lifted_from, objects)
    polys = [(o, footprint(o)) for o in objects if o is not under]

    def hits(xz, tip_y):
        if np.linalg.norm(xz - goal) < into_goal_mm:
            return False
        return any(tip_y - below_marker <= o["height_y"] + 3.0 and polygons_overlap(shape + xz, poly)
                   for o, poly in polys)

    forbid = np.array([hits(xz, wmap.tips[idx, 1]) for idx, xz in enumerate(wmap.xz)], bool)
    cache = {}

    def crossing(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            cache[key] = hits(0.5 * (wmap.xz[a] + wmap.xz[b]), min(wmap.tips[a, 1], wmap.tips[b, 1]))
        return cache[key]
    return forbid, crossing


def carry_forbidden(ctx, wmap, outline, objects, goal, into_goal_mm=30.0, lifted_from=None):
    """Map settings where the carried letter would hit an obstacle (see carry_check)."""
    return carry_check(ctx, wmap, outline, objects, goal, into_goal_mm, lifted_from)[0]


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
    ctx["vision"].forget_tip(expect=ctx["rest"])     # it springs back toward the middle: look there
    end = time.time() + settle_s
    while time.time() < end:
        frame, depth = ctx["vision"].grab()
        ctx["display"].show(frame, ctx["vision"].find_tip(frame, depth))
    reset_live_model(ctx["cables"])
    tip = ctx["vision"].measure_tip(5, ctx["display"].show)
    if tip is None:
        print("    WARNING: can't see the tip after letting go of the cables (hidden, or caught?)")
    elif np.hypot(*(tip["pos"][[0, 2]] - ctx["middle"])) > 30.0:
        print(f"    WARNING: the tip came back to {fmt(tip['pos'])}, not the middle: it may be caught on something")


def drive(ctx, route, target_xyz, tol_mm):
    """Follow a planned route to target_xyz, giving up if the tip gets caught (Watchdog)."""
    args = ctx["args"]
    watch = Watchdog(args.stuck_s, args.stuck_mm, args.attempt_timeout_s)
    if ctx.get("map") is not None and route.get("pulls") is not None:   # chain of recorded settings
        out = play(ctx["cables"], ctx["vision"], ctx["map"], route["pulls"], np.asarray(target_xyz, float),
                   ctx["display"].show, tol_mm=tol_mm, watch=watch, log=lambda *a, **k: None)
    elif ctx.get("map") is not None:               # measured map: play the route back as pulls
        out = follow(ctx["cables"], ctx["vision"], ctx["map"], route["path"], np.asarray(target_xyz, float),
                     ctx["display"].show, tol_mm=tol_mm, watch=watch, log=lambda *a, **k: None)
    else:
        waypoints = densify(route["path"], args.waypoint_step_mm)
        out = move_tip_along(ctx["cables"], ctx["vision"], waypoints, np.asarray(target_xyz, float),
                             ctx["display"].show, tol_mm=tol_mm, adapt=True, watch=watch, log=lambda *a, **k: None)
    if out["reason"] == "stuck":
        out["reason"] = f"stuck: {watch.why}"
    return out


def measure_here(ctx):
    """Measure the cable model where the arm is now (it is bent: the calibration's model is for the middle)."""
    J, _ = probe_model(ctx["cables"], ctx["vision"], ctx["display"].show)
    if J is not None:
        set_live_model(ctx["cables"], J)
        print("    cables here: " + describe_model(J, ctx["cables"].used > 0))


def short_move(ctx, target_xyz, tol_mm, timeout_s):
    """A short move to target_xyz (line up over a magnet, back off): through the map if there is one."""
    args, target = ctx["args"], np.asarray(target_xyz, float)
    watch = Watchdog(args.stuck_s, args.stuck_mm, timeout_s)
    if ctx.get("map") is not None:
        tip = ctx["vision"].measure_tip(3, ctx["display"].show)
        start = tip["pos"][[0, 2]] if tip is not None else target[[0, 2]]
        return follow(ctx["cables"], ctx["vision"], ctx["map"], [start, target[[0, 2]]], target, ctx["display"].show,
                      tol_mm=tol_mm, max_corrections=6, watch=watch, log=lambda *a, **k: None)
    return move_tip_adaptive(ctx["cables"], ctx["vision"], target, ctx["display"].show, tol_mm=tol_mm, max_iter=8,
                             log=lambda *a, **k: None, watch=watch)


def touch(ctx, target_xyz):
    """Line up closely over the magnet, come down onto it (the marker stops tip_length above it)
    and hold a moment, so the tip's magnet takes the letter."""
    short_move(ctx, target_xyz, min(4.0, ctx["args"].success_mm), 15.0)
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
        short_move(ctx, np.array([away[0], tip["pos"][1], away[1]]), 6.0, 20.0)
    now = where_is(ctx, L)
    return now is None or np.hypot(*(now[[0, 2]] - lay_at[[0, 2]])) > 12.0


# ---------------------------------------------------------------- the loop

def cmd_retrieve(cfg, args):
    calib = load_calibration()
    if calib is None:
        sys.exit("No cable_calibration.json yet: run `python main.py calibrate` first.")
    slots = {L: np.asarray(p, float) for L, p in (cfg.get("holder_slots") or {}).items()}
    taught = cfg.get("holder_pull_paths") or {}
    missing = sorted(set(LETTERS) - set(slots) - set(taught))
    if not args.no_place and not args.dry_run and missing:
        sys.exit("The holder's slots aren't taught yet (" + ", ".join(missing) + " missing): run `python main.py "
                 "jog --teach` (drive each letter in by hand) or `holder`, or `retrieve --no-place` to only reach the magnets.")
    for L, t in taught.items():
        if t.get("calibration") != calib.get("created"):
            print(f"Note: slot {L}'s taught path was made with another calibration ({t.get('calibration')}); the pulls "
                  "are counted from each cable's taut point, so it should still fit. Teach it again if it misses.")
    if args.tip_mm is not None:
        cfg["tip_length_mm"] = args.tip_mm
    wmap = None if args.no_map else load_map(calib, pull_limit=args.pull_max_mm)
    if wmap is None and not args.no_map:
        print("No workspace map for this calibration: run `python main.py explore` first (platform cleared) "
              "for much better path following and an honest reach check. Using the adaptive controller.")
    pull_max = args.pull_max_mm if args.pull_max_mm is not None else (
        float(wmap.pull_max.max()) if wmap is not None else 25.0)
    calib = {**calib, "pull_max_mm": [pull_max if s else 0.0 for s in calib["pull_sign"]]}

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
               "rest": rest["pos"].copy(), "middle": rest["pos"][[0, 2]].copy(), "map": wmap,
               # where the tip can go: measured (map), else the linear model with a loose tolerance
               "reachable": wmap.reachable if wmap is not None else reachability(calib, rest["pos"], tol_mm=25.0)}
        if wmap is not None:
            display.region = (wmap.outline(), rest["pos"][1])
            print(f"Workspace map: the tip reaches up to {wmap.reach_mm():.0f} mm from the middle "
                  f"(green outline), cables up to {pull_max:.0f} mm.")
        ctx["slot_marks"] = [(tuple(p), f"slot {L}", COLOURS[L]) for L, p in slots.items()]
        display.marks = list(ctx["slot_marks"])
        ctx["slot_outlines"] = {L: np.asarray(o, float) for L, o in (cfg.get("holder_outlines") or {}).items()
                                if L in slots and len(o) >= 3}
        display.shapes = [(ctx["slot_outlines"][L] + slots[L][[0, 2]], slots[L][1], COLOURS[L]) for L in ctx["slot_outlines"]]
        if slots and len(ctx["slot_outlines"]) < len(slots):
            print("Slots taught without their letters in them: orientation can't be checked. To check it, put each "
                  "letter in its slot and teach it again with `python main.py holder`.")
        print(f"Tip at the middle: {fmt(rest['pos'])}. Tip length past the marker: {cfg['tip_length_mm']:.0f} mm. "
              + ("Holder slots: " + ", ".join(f"{L} {fmt(p)}" for L, p in slots.items()) if slots else "No holder slots."))
        for L, p in slots.items():
            if np.hypot(p[0], p[2]) > cfg["max_radius_mm"] - 10:
                print(f"WARNING: slot {L} is outside the camera's search circle (max_radius_mm "
                      f"{cfg['max_radius_mm']:.0f}): a letter put there can't be seen, so placing it can't be checked.")

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
            plans, missing, too_far = {}, [L for L in todo if L not in letters], []
            for L in todo:
                if L not in letters:
                    print(f"{L}: not found in the camera view")
                    continue
                if wmap is not None:
                    gap = wmap.distance_to_reach(letters[L]["pos"][[0, 2]])
                    if not wmap.reachable(letters[L]["pos"][[0, 2]], tol_mm=args.success_mm + wmap.tolerance_at(
                            letters[L]["pos"][[0, 2]])):
                        print(f"{L}: OUT OF REACH - the tip was never seen closer than {gap:.0f} mm to it "
                              f"(cables up to {pull_max:.0f} mm). Move it inside the green outline, or explore "
                              "with more cable (`explore --pull-max-mm 35`).")
                        too_far.append(L)
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
                stuck = [L for L in todo if L not in missing and L not in too_far]
                print((f"No route left to try for {', '.join(stuck)}. " if stuck else "") +
                      (f"Out of reach: {', '.join(too_far)}. " if too_far else "") +
                      (f"Not found: {', '.join(missing)} (in the grey circle? taught? see `detect`). " if missing else "")
                      + "Stopping.")
                break
            L = min(plans, key=lambda k: plans[k][0][0]["cost"])
            route = plans[L][0][0]
            show_plans(display, plans, chosen=L, marks=ctx["slot_marks"])
            print(f"\nPlan: " + " | ".join(
                (f"{k} map route {r[0]['length']:.0f} mm, in from {r[0]['angle_deg']:.0f} deg (cost {r[0]['cost']:.0f})"
                 if r[0].get("graph") else
                 f"{k} best side {r[0]['angle_deg']:.0f} deg, {r[0]['length']:.0f} mm (cost {r[0]['cost']:.0f}, "
                 f"{len(r)} side(s) left)") for k, (r, _) in plans.items()))

            if args.dry_run:
                for k in plans:
                    turn = orientation(ctx, k, letters[k])
                    if turn is not None:
                        print(f"  {k} lies turned {turn[0]:+.0f} deg from its slot"
                              + (" - turn it on the table, the arm can't" if abs(turn[0]) > args.yaw_tol_deg else ""))
                    h = plans[k][0][0].get("end_height")
                    if h is not None:
                        print(f"  {k}: lowest tip the map has near it: {h:+.0f} mm from touching")
                display.lines = [f"DRY RUN: would go to {L} first (side {route['angle_deg']:.0f} deg)  q = quit",
                                 "cyan = chosen route, grey = other candidates, dots = approach points"]
                end = time.time() + args.dry_run_s
                while time.time() < end:
                    frame, depth = vision.grab()
                    display.show(frame, vision.find_tip(frame, depth))
                break

            turn = orientation(ctx, L, letters[L])
            if turn is not None:
                print(f"{L} lies turned {turn[0]:+.0f} deg from its slot (shape match {turn[1]:.2f})"
                      + (f" - WARNING: more than {args.yaw_tol_deg:.0f} deg; the arm can't turn a letter, so turn it on "
                         "the table to match its slot or it won't go in" if abs(turn[0]) > args.yaw_tol_deg else ""))
            if route.get("end_height") is not None and route["end_height"] > args.touch_height_mm:
                print(f"{L}: the lowest the map saw the tip near it is {route['end_height']:.0f} mm above touching "
                      "(the arm rises as it bends out): the magnet may not reach it from here")
            attempts[L] += 1
            target = letters[L]["pos"]
            display.target_px, display.target_pos = letters[L]["px"], target
            display.lines = [status, f"{L}: attempt {attempts[L]}, from side {route['angle_deg']:.0f} deg"]
            print(f"{L}: attempt {attempts[L]} " + ("straight in (no side route left), " if route.get("direct") else "")
                  + f"from side {route['angle_deg']:.0f} deg, "
                  f"{len(route['path']) - 2} corner(s), {route['length']:.0f} mm, magnet at {fmt(target)}")
            started = time.time()
            out = drive(ctx, route, target, args.success_mm)
            stage, ok = "reach", out["reached"]
            if ok:
                touch(ctx, target)
                tip_now = ctx["vision"].measure_tip(3, display.show)
                if tip_now is not None:
                    above = tip_now["pos"][1] - (target[1] + cfg["tip_length_mm"])
                    if above > args.touch_height_mm:
                        print(f"  {L}: the tip is {above:.0f} mm above touching the magnet (it can't get lower here)")
                if args.no_place:
                    print(f"  {L}: reached its magnet ({out['error_mm']:.1f} mm off)")
                    placed.add(L)
                else:
                    stage = "grab"
                    ok = grabbed(ctx, L, target)
                    out["reason"] = "picked up" if ok else "letter didn't come along"
                    if ok:
                        stage = "place"
                        ok, out = place(ctx, L, slots.get(L), letters, objects, placed)
                        if ok:
                            placed.add(L)
            rows.append(log_row(L, attempts[L], stage, route, ok, out, time.time() - started, target))
            print(f"  {L}: {'OK' if ok else 'FAILED'} at {stage} ({out['reason']})")
            if not ok:
                failed[L].append(failure(route, out))
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


def failure(route, out):
    """What to avoid next time: the side it came from, where the tip ended up, the approach point."""
    tip = out.get("tip")
    spot = None if tip is None or not np.all(np.isfinite(tip)) else np.asarray(tip, float)[[0, 2]]
    return {"angle": route["angle_deg"], "spot": spot, "approach": np.asarray(route["approach"], float)}


def place(ctx, L, slot, letters, objects, placed):
    """Carry the picked-up letter to its slot (planned like a letter: the holder is the support),
    put it down, back off to the middle, and check it is there. Returns (ok, outcome)."""
    args = ctx["args"]
    taught = (ctx["cfg"].get("holder_pull_paths") or {}).get(L)
    if taught:
        return place_taught(ctx, L, slot, taught, letters)
    turn = orientation(ctx, L, letters.get(L, {}))
    if turn is not None:
        print(f"    {L} is carried turned {turn[0]:+.0f} deg from its slot"
              + (f": WARNING, more than {args.yaw_tol_deg:.0f} deg, it probably won't go in (the arm can't turn it)"
                 if abs(turn[0]) > args.yaw_tol_deg else ""))
    if ctx.get("map") is None:
        measure_here(ctx)                            # without a map: the model for this bent pose
    out = {"reached": False, "reason": "no route to the slot", "error_mm": np.nan, "tip": None}
    tried = []
    for _ in range(args.place_tries):
        tip = ctx["vision"].measure_tip(3, ctx["display"].show)
        start = tip["pos"][[0, 2]] if tip is not None else ctx["middle"]
        routes, _ = plan(ctx, start, slot, letters, objects, skip={L} | placed, failed=tried, carrying=True,
                         start_pulls=ctx["cables"].pulls.copy(), carried=letters.get(L))
        if not routes:
            break
        route = routes[0]
        ctx["display"].path, ctx["display"].target_pos = (route["path"], slot[1]), slot
        print(f"    carrying {L} to its slot from side {route['angle_deg']:.0f} deg"
              + (f" (no way keeps it fully clear: it may brush an obstacle {route['brushes']} time(s))"
                 if route.get("brushes") else ""))
        out = drive(ctx, route, slot, args.success_mm)
        if out["reached"]:
            break
        tried.append(failure(route, out))
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


def place_taught(ctx, L, slot, taught, letters):
    """Put the picked-up letter in its slot by replaying, from the middle, the cable poses taught
    with `jog --teach` (no tip tracking on the holder), let go, and check it is there."""
    print(f"    carrying {L} to its slot along the taught path ({len(taught['pulls'])} pose(s))")
    back_to_middle(ctx)                              # the taught path starts from the middle
    ok, why = replay(ctx["cables"], ctx["vision"], ctx["display"], taught["pulls"])
    out = {"reached": ok, "reason": why, "error_mm": np.nan, "tip": None}
    if not ok:
        return False, out
    hold(ctx, 0.8)
    back_to_middle(ctx)                              # the slot keeps the letter, the magnet lets go
    if slot is None:
        return True, {**out, "reason": "put in along the taught path (slot position not taught: not checked)"}
    now = where_is(ctx, L)
    if now is not None and np.hypot(*(now[[0, 2]] - slot[[0, 2]])) < ctx["args"].slot_tol_mm:
        return True, {**out, "reason": "in its slot"}
    if now is not None:
        letters[L]["pos"] = now
        return False, {**out, "reason": f"dropped at {fmt(now)}, not in the slot"}
    return False, {**out, "reason": "not seen after letting go (still on the magnet?)"}


def replay(cables, vision, display, sequence, step_mm=2.5, max_load_ma=1500.0):
    """Move the cables through a taught sequence of pulls, no motor changing more than step_mm per
    move. The camera only shows; it doesn't steer. Stops if a motor's current rises by
    max_load_ma (caught on something). Returns (ok, reason)."""
    limit = list(cables.calib["pull_max_mm"])
    cables.calib = {**cables.calib, "pull_max_mm": [TEACH_PULL_MAX_MM if s else 0.0 for s in cables.calib["pull_sign"]]}
    try:
        baseline = np.abs(cables.read_currents())
    except Exception:
        baseline = np.zeros(4)
    try:
        for target in sequence:
            target, start = np.asarray(target, float), cables.pulls.copy()
            n = max(1, int(np.ceil(np.abs(target - start).max() / step_mm)))
            for k in range(1, n + 1):
                cables.set_pulls(start + (target - start) * k / n)
                frame, depth = vision.grab()
                display.show(frame, vision.find_tip(frame, depth))
                load = _cable_load(cables, baseline)
                if load > max_load_ma:
                    return False, f"motor current +{load:.0f} mA on the way: caught on something"
        return True, "reached the taught pose"
    finally:
        cables.calib = {**cables.calib, "pull_max_mm": limit}


def release(cables, vision, display):
    """Let all cables go in a few steps (the arm goes back to the middle)."""
    for frac in (0.66, 0.33, 0.0):
        cables.set_pulls(cables.pulls * frac)
        frame, depth = vision.grab()
        display.show(frame, vision.find_tip(frame, depth))
    vision.forget_tip()


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
    (from the depth camera) is saved as holder_slots in config.json."""
    camera = open_camera_or_exit()
    vision = Vision(cfg, camera)
    display = Display(vision)
    slots = {L: list(p) for L, p in (cfg.get("holder_slots") or {}).items()}
    outlines = {L: o for L, o in (cfg.get("holder_outlines") or {}).items() if L in slots}
    state = {"click": None, "depth": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.setMouseCallback(WINDOW, on_mouse)
    print("Best: put each letter in its slot, click it, press n / u / s (learns where its magnet sits and which\n"
          "way it faces). Otherwise click the middle of an empty slot and press the key (position only).\n"
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
                # best: the letter itself sitting in its slot -> its magnet position and its orientation
                found, _ = vision.detect_letters(frame, depth)
                d = found.get(L)
                on_it = d is not None and (np.hypot(*(d["px"] - np.array([u, v]))) < 40 or cv2.pointPolygonTest(
                    d["contour"].astype(np.int32), (float(u), float(v)), True) > -10)
                if on_it and d["pos"] is not None:
                    slots[L] = [round(float(c), 1) for c in d["pos"]]
                    out = vision.letter_outline(d)
                    if out is not None:
                        outlines[L] = np.round(out, 1).tolist()
                    print(f"Slot {L}: {fmt(d['pos'])}, from the {L} sitting in it (position and orientation)")
                    continue
                outlines.pop(L, None)
                window = depth[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4]
                d = window[window > 0]
                pos = (vision.pixel_to_robot(u, v, float(np.median(d))) if len(d) >= 5
                       else vision.pixel_to_plane(u, v, cfg["stage_y_mm"]))
                if pos is None:
                    print("Couldn't get a position there: click on the holder.")
                    continue
                slots[L] = [round(float(c), 1) for c in pos]
                print(f"Slot {L}: {fmt(pos)}" + ("   WARNING: outside the camera's search circle, move the holder "
                                                 "closer to the middle" if np.hypot(pos[0], pos[2]) > cfg["max_radius_mm"] - 10 else ""))
            elif key == ord("c"):
                slots, outlines = {}, {}
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()
    save_config_value("holder_slots", slots)
    save_config_value("holder_outlines", outlines)
    print(f"Saved holder_slots ({', '.join(sorted(slots)) or 'none'}) to config.json")


# ---------------------------------------------------------------- teach the holder by hand

TEACH_PULL_MAX_MM = 60.0


def cmd_teach(cfg, args):
    """Drive the arm by keyboard (cable pulls from the calibration's taut points) until the letter
    on the magnet sits in its slot, keeping poses on the way (a), then save that sequence for the
    slot (Enter, then n / u / s). `retrieve` replays it to put that letter in."""
    calib = load_calibration()
    if calib is None:
        sys.exit("No cable_calibration.json yet: run `python main.py calibrate` first.")
    unused = [j for j in range(4) if calib["pull_sign"][j] == 0]
    if unused:
        print(f"WARNING: {', '.join(f'M{j}' for j in unused)} UNUSED in the calibration, so it can't move here. "
              "Calibrate again (e.g. `calibrate --pull-signs=1,1,1,1 --push M0`) if the holder needs it.")
    paths = dict(cfg.get("holder_pull_paths") or {})
    print("Teach each slot: put the letter on the magnet, then drive the arm until the letter sits in its slot.\n"
          "  0-3 select a motor   w / s = pull +1 / -1 mm   W / S = +5 / -5 mm\n"
          "  a = keep this pose as a step on the way (e.g. above the holder's edge)   x = forget the steps\n"
          "  Enter, then n / u / s = save the steps + this pose for that slot\n"
          "  r = let all cables go (back to the middle)   p, then n / u / s = replay a saved slot from the middle\n"
          "  q = quit (each slot is saved to config.json as soon as you save it)")
    camera = open_camera_or_exit()
    vision = Vision(cfg, camera)
    display = Display(vision)
    cables = Cables(calib)
    steps_mm = {ord("w"): 1.0, ord("s"): -1.0, ord("W"): 5.0, ord("S"): -5.0}
    letter_keys = {ord("n"): "N", ord("u"): "U", ord("s"): "S"}
    try:
        if not cables.open():
            print("Could not connect to the motors (power on? USB plugged in? EMIO Labs closed?).")
            return
        align_to_calibration(cables)
        limit = [TEACH_PULL_MAX_MM if s else 0.0 for s in calib["pull_sign"]]
        cables.calib = {**cables.calib, "pull_max_mm": limit}
        cables.set_pulls(np.zeros(4))
        motor, steps, waiting = 0, [], None          # waiting: "save" / "play" (after Enter / p)
        while True:
            frame, depth = vision.grab()
            tip = vision.find_tip(frame, depth)
            prompt = {"save": "save for which slot?  n / u / s", "play": "replay which slot?  n / u / s"}
            display.lines = [f"M{motor} selected | pulls " + "  ".join(f"M{j} {cables.pulls[j]:.0f}" for j in range(4)) + " mm",
                             f"{len(steps)} step(s) kept | saved: {', '.join(sorted(paths)) or 'none'}",
                             prompt.get(waiting, "0-3 w/s W/S move | a step | Enter save | x forget | r release | p replay | q quit")]
            key = display.show(frame, tip)
            if key == 255:
                continue
            if waiting is not None:
                mode, waiting, L = waiting, None, letter_keys.get(key)
                if L is None:
                    continue
                if mode == "save":
                    seq = [[round(float(v), 1) for v in p] for p in steps + [cables.pulls.copy()]]
                    paths[L] = {"pulls": seq, "calibration": calib.get("created"),
                                "tip": None if tip is None else [round(float(v), 1) for v in tip["pos"]]}
                    save_config_value("holder_pull_paths", paths)
                    print(f"Saved slot {L}: {len(seq)} pose(s), ending at pulls {seq[-1]} mm")
                    steps = []
                elif L in paths:
                    release(cables, vision, display)
                    ok, why = replay(cables, vision, display, paths[L]["pulls"])
                    print(f"Replayed slot {L}: {why}")
                else:
                    print(f"No path saved for {L} yet.")
                continue
            if ord("0") <= key <= ord("3"):
                motor = key - ord("0")
                if limit[motor] == 0:
                    print(f"M{motor} is UNUSED in the calibration: it won't move.")
            elif key in steps_mm:
                p = cables.pulls.copy()
                p[motor] = float(np.clip(p[motor] + steps_mm[key], 0.0, limit[motor]))
                cables.set_pulls(p)
            elif key == ord("a"):
                steps.append(cables.pulls.copy())
                print(f"Step {len(steps)}: pulls {[round(float(v), 1) for v in steps[-1]]} mm")
            elif key in (13, 10):
                waiting = "save"
            elif key == ord("p"):
                waiting = "play"
            elif key == ord("x"):
                steps = []
            elif key == ord("r"):
                release(cables, vision, display)
                steps = []
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError) as e:
        print(f"Stopped: {e}")
    finally:
        cables.close()
        camera.close()
        cv2.destroyAllWindows()
    print(f"Taught slots: {', '.join(sorted(paths)) or 'none'} (holder_pull_paths in config.json)")


# ---------------------------------------------------------------- map the workspace

def cmd_explore(cfg, args):
    """Map where the tip really goes for combinations of cable pulls (see workspace_map.explore)."""
    from workspace_map import explore
    calib = load_calibration()
    if calib is None:
        sys.exit("No cable_calibration.json yet: run `python main.py calibrate` first.")
    unused = [j for j in range(4) if calib["pull_sign"][j] == 0]
    if unused:
        sys.exit(f"The calibration has no cable for {', '.join(f'M{j}' for j in unused)} (UNUSED: the tip didn't move "
                 "while it searched), so explore couldn't reach that side of the workspace. Its cable is probably "
                 "slack (the spools unwind while the motors are off): run `python main.py calibrate "
                 "--pull-signs=1,1,1,1` again (it now searches 50 mm), then explore.")
    print(f"Explore pulls each pair of neighbouring cables through a grid (every {args.step_mm:.0f} mm) as far as "
          f"the arm goes: until the tip stops moving, a motor can't pull further, its current rises by "
          f"{args.max_ma:.0f} mA, or {args.pull_max_mm:.0f} mm. It records where the tip goes: a few minutes.\n"
          "Set up exactly as for the run: platform height, OBSTACLES AND HOLDER IN PLACE (the map learns how the\n"
          "arm moves with them), but TAKE THE LETTERS OFF (the magnet would pick them up). Hands out of view.\n"
          "Watch the arm: q in the camera window (or Ctrl+C) stops and lets the cables go.")
    if input("Letters off, obstacles in place? Start? [y/N] ").strip().lower() != "y":
        return
    camera = open_camera_or_exit()
    vision = Vision(cfg, camera)
    display = Display(vision)
    cables = Cables(calib)
    try:
        if not cables.open():
            print("Could not connect to the motors (power on? USB plugged in? EMIO Labs closed?).")
            return
        align_to_calibration(cables)
        cables.calib = {**cables.calib, "pull_max_mm": [args.pull_max_mm if s else 0.0 for s in calib["pull_sign"]]}
        display.lines = ["exploring the workspace...  q = stop"]
        wmap = explore(cables, vision, display.show, args.pull_max_mm, step_mm=args.step_mm, max_load_ma=args.max_ma,
                       only_motors=args.only, push=args.push)
        display.region = (wmap.outline(), wmap.rest[1])
        display.lines = [f"reachable area (green): up to {wmap.reach_mm():.0f} mm from the middle"]
        hold({"vision": vision, "display": display}, 4.0)
    except KeyboardInterrupt:
        print("Stopped (nothing saved).")
    except (RuntimeError, ValueError) as e:
        print(f"Stopped: {e}")
    finally:
        cables.close()
        camera.close()
        cv2.destroyAllWindows()
