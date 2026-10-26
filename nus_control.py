"""
Cable-driven manipulator control for the N-U-S retrieval task.

Pull coordinates: p_j = mm of cable that motor j has pulled in beyond its "just taut" angle.
Cables can only pull, so p_j >= 0.
    angle_j = taut_angle_j + pull_sign_j * p_j / PULLEY_RADIUS_MM

Model, measured by `calibrate`:  tip ~= tip_taut + J @ p,
with J the 3x4 Jacobian (mm of tip motion per mm of cable). It is only a local, linear
approximation, so positioning runs closed-loop on the camera (Lecture 5: Jacobian-based
task-space control), which also absorbs camera-calibration errors because the tip and the
letters are measured by the same camera.

Raising the tip: tendons can't shorten the arm (pulling all cables together only stiffens it), so
the tip only rises by bending, which also moves it sideways. raise_tip finds the bend that puts
the tip highest by measuring, with the camera, how each cable changes the tip's height; the moves
that follow keep that height (3D model with the height as a goal) and lower_onto comes down onto
the letter at the end.
"""

import json
import math
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from nus_vision import fmt

HERE = Path(__file__).resolve().parent
CALIB_PATH = HERE / "cable_calibration.json"

PULLEY_RADIUS_MM = 20.0        # emioapi: _length_to_rad = 1/20
PROFILE_VELOCITY = 60          # Dynamixel units of 0.229 rev/min -> ~14 rev/min (slow)
MAX_TRAVEL_RAD = 3.5           # never command a motor more than this (~70 mm of cable) from where it started
MAX_DRIFT_RAD = 0.5            # spool moved more than this (~10 mm) since calibration -> recalibrate

# Dynamixel X-series control table (XM430-W210)
DXL_IDS = (0, 1, 2, 3)
ADDR_OPERATING_MODE, ADDR_TORQUE_ENABLE, ADDR_GOAL_POSITION, ADDR_PRESENT_POSITION = 11, 64, 116, 132
ADDR_PRESENT_CURRENT = 126     # 2 bytes, signed
CURRENT_UNIT_MA = 2.69
EXTENDED_POSITION_MODE = 4     # multi-turn: +-256 turns, no single-turn position limits

# calibration
SEARCH_STEP_RAD = 0.05         # 1 mm of cable per step
SEARCH_MAX_MM = 30.0           # give up on a direction after this much cable
TAUT_SHIFT_PX = 5.0            # tip shift (pixels) that means "this cable is now pulling"
CONFIRM_PULL_MM = 4            # ...if pulling this much further moves the tip on by TAUT_SHIFT_PX again
JACOBIAN_PULL_MM = 5.0         # test pull used to measure each Jacobian column (halved if the tip is lost)
PULL_MAX_MM = 15.0             # never pull a cable more than this beyond taut (raise once you trust it)

# raising the tip (raise_tip / lower_onto)
RAISE_PROBE_MM = 3.0           # test pull per cable when measuring how it changes the tip's height
RAISE_STEP_MM = 4.0            # most any one cable changes per raising step
RAISE_MAX_LOAD_MA = 300.0      # stop when a motor draws this much more current than before raising (~20 N on the cable)
RAISE_MIN_GAIN_MM = 1.0        # a step "raised the tip" if it got at least this much higher (depth noise ~1 mm)


def _signed32(value):
    return value - (1 << 32) if value >= (1 << 31) else value


class Cables:
    def __init__(self, calib=None, max_travel_rad=MAX_TRAVEL_RAD):
        from emioapi import EmioMotors
        self.motors = EmioMotors()
        self.calib = calib
        self.pulls = np.zeros(4)
        self.reference = None          # angles at open(): the travel limit is measured from here
        self.max_travel_rad = max_travel_rad
        self.is_open = False

    def open(self):
        if self.motors.findAndOpen() < 0:
            return False
        self.is_open = True
        self._enter_extended_position_mode()
        self.motors.max_velocity = [PROFILE_VELOCITY] * 4
        self.reference = self.read_angles()
        return True

    def _enter_extended_position_mode(self):
        """A cable spool can need more than one turn, but in single-turn Position mode the servo
        refuses any goal past the turn boundary (your motors 2 and 3 sat just past it).
        Each servo: torque off -> multi-turn mode -> goal = present position -> torque on,
        so enabling torque can't make it jump. emioapi switches back to Position mode on its next open."""
        from dynamixel_sdk import COMM_SUCCESS
        mg = self.motors._mg                      # emioapi internals: the Dynamixel port and packet handler
        port, ph = mg.portHandler, mg.packetHandler

        def check(what, dxl_id, result, error):
            if result != COMM_SUCCESS or error != 0:
                raise RuntimeError(f"motor {dxl_id}: {what} failed ({ph.getTxRxResult(result)} "
                                   f"{ph.getRxPacketError(error)})")

        for dxl_id in DXL_IDS:
            check("torque off", dxl_id, *ph.write1ByteTxRx(port, dxl_id, ADDR_TORQUE_ENABLE, 0))
            check("set multi-turn mode", dxl_id,
                  *ph.write1ByteTxRx(port, dxl_id, ADDR_OPERATING_MODE, EXTENDED_POSITION_MODE))
            present, result, error = ph.read4ByteTxRx(port, dxl_id, ADDR_PRESENT_POSITION)
            check("read position", dxl_id, result, error)
            check("hold position", dxl_id, *ph.write4ByteTxRx(port, dxl_id, ADDR_GOAL_POSITION, present))
            check("torque on", dxl_id, *ph.write1ByteTxRx(port, dxl_id, ADDR_TORQUE_ENABLE, 1))

    def close(self):
        if self.is_open:
            self.motors.close()                          # disables torque: the arm goes slack
            self.is_open = False

    def read_angles(self):
        """emioapi reads the signed position register as unsigned (a position just below 0 comes
        back as ~4.29e9 pulses, i.e. about -6.6 million rad), so convert to signed first."""
        pulses = [_signed32(p) for p in self.motors._mg.getCurrentPosition()]
        return np.array(self.motors.pulseToRad(pulses), float)

    def move_angles(self, angles, timeout=3.0):
        angles = [float(a) for a in angles]
        travel = np.abs(np.array(angles) - self.reference)
        if np.any(travel > self.max_travel_rad):
            bad = [j for j in range(4) if travel[j] > self.max_travel_rad]
            raise ValueError(f"motor(s) {bad}: target is more than {self.max_travel_rad:.1f} rad "
                             f"(~{self.max_travel_rad * PULLEY_RADIUS_MM:.0f} mm of cable) from the start position: "
                             f"{np.round(angles, 3)}")
        self.motors.angles = angles
        self.goal_angles = np.array(angles)
        self._wait(timeout)

    def lag_mm(self):
        """How far (mm of cable) each motor is from where it was last told to go: a motor that
        can't get there is pulling as hard as it can (the cable or the arm won't give)."""
        if getattr(self, "goal_angles", None) is None:
            return np.zeros(4)
        return np.abs(self.goal_angles - self.read_angles()) * PULLEY_RADIUS_MM

    def _wait(self, timeout, settle=0.15):
        time.sleep(0.05)
        end = time.time() + timeout
        while time.time() < end:
            moving = self.motors.moving
            if moving is not None and not any(moving):
                break
            time.sleep(0.03)
        time.sleep(settle)

    def pulls_to_angles(self, pulls):
        c = self.calib
        return [c["taut_angle"][j] + c["pull_sign"][j] * pulls[j] / PULLEY_RADIUS_MM for j in range(4)]

    def set_pulls(self, pulls):
        pulls = np.clip(np.asarray(pulls, float), 0.0, self.calib["pull_max_mm"])
        self.move_angles(self.pulls_to_angles(pulls))
        self.pulls = pulls
        return pulls

    @property
    def used(self):
        """1 for each motor with a cable, 0 for unused ones."""
        return (np.array(self.calib["pull_sign"]) != 0).astype(float)

    def read_currents(self):
        """Motor currents in mA (NaN where a read fails). A motor holding a tighter cable draws more."""
        from dynamixel_sdk import COMM_SUCCESS
        mg = self.motors._mg
        currents = []
        for dxl_id in DXL_IDS:
            value, result, error = mg.packetHandler.read2ByteTxRx(mg.portHandler, dxl_id, ADDR_PRESENT_CURRENT)
            signed = value - (1 << 16) if value >= (1 << 15) else value
            currents.append(signed * CURRENT_UNIT_MA if result == COMM_SUCCESS else math.nan)
        return np.array(currents)


def load_calibration():
    return json.loads(CALIB_PATH.read_text()) if CALIB_PATH.exists() else None


def align_to_calibration(cables, log=print):
    """The servos report spool angles only within one turn after power-up or a mode change, so
    shift each calibrated taut angle by whole turns to match the spool's current turn. Then refuse
    if a spool has really moved since calibration (e.g. the arm was pulled while unpowered):
    the taut points would be wrong and cables could be over-pulled."""
    now = cables.read_angles()
    taut = np.array(cables.calib["taut_angle"], float)
    taut += np.round((now - taut) / (2 * math.pi)) * 2 * math.pi
    used = np.array(cables.calib["pull_sign"]) != 0
    taut[~used] = now[~used]                 # unused motors (no cable) just stay where they are
    drift = now - taut
    log("Spool drift since calibration: " + ", ".join(
        f"M{j} {drift[j] * PULLEY_RADIUS_MM:+.1f} mm" if used[j] else f"M{j} unused" for j in range(4)))
    if np.any(np.abs(drift) > MAX_DRIFT_RAD):
        raise RuntimeError("A cable spool has moved since calibration (more than "
                           f"{MAX_DRIFT_RAD * PULLEY_RADIUS_MM:.0f} mm). Run `python nus_main.py calibrate` again.")
    cables.calib = {**cables.calib, "taut_angle": taut.tolist()}


def solve_pulls(J, d, p_max):
    """Least-squares p >= 0 (cables only pull), p_j <= p_max_j, with J @ p ~= d.
    Only 4 motors, so simply try every subset of active cables."""
    n = J.shape[1]
    p_max = np.asarray(p_max, float)
    best_p, best_r = np.zeros(n), float(np.linalg.norm(d))
    for k in (1, 2, 3):
        for cols in combinations(range(n), k):
            cols = list(cols)
            sol = np.linalg.lstsq(J[:, cols], d, rcond=None)[0]
            if np.any(sol < 0):
                continue
            p = np.zeros(n)
            p[cols] = np.minimum(sol, p_max[cols])
            r = float(np.linalg.norm(J @ p - d))
            if r < best_r - 1e-6 or (abs(r - best_r) <= 1e-6 and p.sum() < best_p.sum()):
                best_p, best_r = p, r
    return best_p, best_r


def calibrate(cables, vision, on_frame, first_direction=-1, pull_signs=None, search_mm=SEARCH_MAX_MM, log=print):
    """Find each cable's pull direction and 'just taut' angle, then measure the Jacobian.
    Expects the arm at rest with all cables slack: hanging straight, or lying bent on a raised
    platform (the model is then measured around that bent pose, which is where it is used).
    pull_signs: known pull direction per motor (e.g. from `jog`); then only that direction is searched,
    so no cable is ever unwound."""
    start = cables.read_angles()
    base = vision.measure_tip(10, on_frame)
    if base is None:
        raise RuntimeError("Green tip not visible. Check the camera view and the tip_hsv setting.")
    off_axis = math.hypot(base["pos"][0], base["pos"][2])
    log(f"Rest tip at {fmt(base['pos'])} (pixel {base['px'].astype(int).tolist()}), "
        f"{off_axis:.0f} mm off the robot axis")
    if off_axis > 15:
        # Expected when the raised platform makes the arm rest bent. (Green objects elsewhere in the
        # room are already rejected by find_tip's 3D reachability filter.)
        log("Note: the tip rests off-centre. That's fine if the arm lies bent on the raised platform; "
            "otherwise a cable may already be pulling and should be slackened first.")

    # 1. pull direction and taut angle, one motor at a time. A motor whose cable never moves the
    #    tip (e.g. no cable attached: a 3-tendon arm uses only 3 of the 4 motors) is marked unused.
    cables.max_travel_rad = max(cables.max_travel_rad, search_mm / PULLEY_RADIUS_MM + 0.1)
    pull_sign, taut = [0] * 4, [float(a) for a in start]
    for j in range(4):
        if pull_signs and pull_signs[j] == 0:
            log(f"Motor {j}: marked unused (--pull-signs 0)")
            continue
        directions = (pull_signs[j],) if pull_signs else (first_direction, -first_direction)
        trials = [_try_direction(cables, vision, on_frame, start, base, j, sign, search_mm, log) for sign in directions]
        # a PULLING cable keeps moving the tip further out (and the motor works harder); a cable being
        # LET OUT moves it a bit too (the other cables pull it over, an arm that was pressed relaxes),
        # then stops once it is slack: compare how much each direction still moves the tip 4-8 mm on
        good = [t for t in trials if t["k"] is not None and t["growth_px"] >= TAUT_SHIFT_PX]
        for t in trials:
            if t["k"] is not None:
                rise = "" if t["current_rise"] is None else f", motor current {t['current_rise']:+.0f} mA"
                log(f"Motor {j} direction {t['sign']:+d}: tip shifted after {t['k']} mm, then moved "
                    f"{t['growth_px']:.0f} px more over the next 4 mm{rise}")
        if not good:
            if any(t["k"] is not None for t in trials):
                log(f"Motor {j}: the tip shifted but didn't keep moving in any direction tried (only let out?)")
            log(f"Motor {j}: UNUSED - no direction kept moving the tip. Either no cable is attached to it, or "
                f"its cable has more slack than {search_mm:.0f} mm (wind it in with `python nus_main.py jog`, "
                "then calibrate again).")
            continue
        currents = [t["current_rise"] for t in good]
        if len(good) == 2 and None not in currents and abs(currents[0] - currents[1]) > 30.0:
            pick = max(good, key=lambda t: t["current_rise"])       # pulling tightens: more current
        else:
            pick = max(good, key=lambda t: t["growth_px"])
        pull_sign[j] = pick["sign"]
        taut[j] = float(start[j] + pick["sign"] * (pick["k"] - 1) * SEARCH_STEP_RAD)
        log(f"Motor {j}: pulls with direction {pick['sign']:+d}, taut after {pick['k'] - 1} mm "
            f"(angle {taut[j]:+.3f} rad)")

    used = [j for j in range(4) if pull_sign[j] != 0]
    if len(used) < 2:
        raise RuntimeError(f"Only {len(used)} motor(s) move the tip ({used}): need at least 2. "
                           "Check the cables are attached and not slack (use `jog`).")
    if len(used) < 3:
        log(f"WARNING: only motors {used} move the tip. Cables can only pull, so 2 cables can't reach "
            "every direction; some letter positions will be out of reach.")
    pull_max = [PULL_MAX_MM if j in used else 0.0 for j in range(4)]
    cables.calib = {"pull_sign": pull_sign, "taut_angle": taut, "pull_max_mm": pull_max}

    # 2. Jacobian: pull each cable a little from the all-taut pose and watch the tip. Each move is
    #    measured from where the tip is just before that pull: a tip resting on the platform doesn't
    #    always spring back when the previous cable is released (friction), and measuring every
    #    cable from the first position then mixes the cables up.
    cables.set_pulls(np.zeros(4))
    vision.forget_tip(expect=base["pos"])
    taut_tip = vision.measure_tip(10, on_frame)
    if taut_tip is None:
        raise RuntimeError("Lost the tip at the taut pose.")
    J = np.zeros((3, 4))                     # unused motors keep a zero column
    base = taut_tip
    for j in used:
        m = None
        for pull_mm in (JACOBIAN_PULL_MM, JACOBIAN_PULL_MM / 2):
            base = vision.measure_tip(6, on_frame) or base
            p = np.zeros(4)
            p[j] = min(pull_mm, pull_max[j])
            cables.set_pulls(p)
            m = vision.measure_tip(6, on_frame) or vision.measure_tip(6, on_frame)   # one retry
            cables.set_pulls(np.zeros(4))
            if m is not None:
                break
            log(f"Motor {j}: lost the tip after pulling {p[j]:.1f} mm, trying a smaller pull")
        if m is None:
            raise RuntimeError(f"Lost the tip while pulling cable {j}: it may leave the camera view or be "
                               "hidden behind the arm. Watch the camera window during that step.")
        J[:, j] = (m["pos"] - base["pos"]) / p[j]
        horiz = J[[0, 2], j]
        log(f"Motor {j}: 1 mm of cable moves the tip {np.linalg.norm(horiz):.2f} mm sideways "
            f"(direction {math.degrees(math.atan2(horiz[1], horiz[0])):+.0f} deg in the x-z plane), "
            f"{J[1, j]:+.2f} mm up")
        if np.linalg.norm(horiz) < 0.2:
            log(f"WARNING: cable {j} barely moves the tip; it may not be taut or may be slipping.")
        back = vision.measure_tip(6, on_frame)
        if back is not None and np.linalg.norm((back["pos"] - base["pos"])[[0, 2]]) > 4.0:
            log(f"Motor {j}: the tip didn't come back by {np.linalg.norm((back['pos'] - base['pos'])[[0, 2]]):.0f} mm "
                "after the cable was released (friction with the platform?). The search re-measures the "
                "cables at every letter, but a freely moving tip calibrates better.")
    # the cables should pull the tip in clearly different directions, all round: two pulling the same
    # way with a side that nothing pulls toward means one of them is being let out, not pulled
    angle = {j: math.degrees(math.atan2(J[2, j], J[0, j])) for j in used if np.linalg.norm(J[[0, 2], j]) > 0.2}
    ordered = sorted(angle.values())
    gaps = [((b - a) % 360.0) for a, b in zip(ordered, ordered[1:] + ordered[:1])] if len(ordered) >= 2 else []
    if len(ordered) >= 3 and max(gaps) > 150.0:
        for a, b in combinations(angle, 2):
            if abs((angle[a] - angle[b] + 180.0) % 360.0 - 180.0) < 40.0:
                log(f"WARNING: motors {a} and {b} both pull the tip toward about {angle[a]:.0f} deg, and no cable "
                    f"pulls toward the other side ({max(gaps):.0f} deg gap): one of them is probably turning the "
                    "wrong way (letting its cable out). Check with `python nus_main.py jog` which way tightens "
                    "each cable, then calibrate with --pull-signs.")
    for a, b in combinations(used, 2):       # two cables that look identical = a measurement went wrong
        if np.linalg.norm(J[:, a] - J[:, b]) < 0.05 * max(np.linalg.norm(J[:, a]), 1e-6) + 1e-3:
            log(f"WARNING: motors {a} and {b} measured exactly the same: the tip probably didn't move for "
                f"one of them. Check both cables are taut and the tip can move, then calibrate again.")

    calib = {**cables.calib, "tip_taut": taut_tip["pos"].tolist(), "J": J.tolist(), "used_motors": used,
             "created": datetime.now().isoformat(timespec="seconds")}
    cables.calib = calib
    CALIB_PATH.write_text(json.dumps(calib, indent=2))
    log(f"Saved {CALIB_PATH.name}")
    return calib


def _try_direction(cables, vision, on_frame, start, base, j, sign, search_mm, log):
    """Turn motor j in `sign` direction 1 mm at a time until the tip shifts (TAUT_SHIFT_PX), then
    4 and 8 mm further, and put it back. Returns {"sign", "k": mm at the shift (None: never),
    "growth_px": how far the tip moved between +4 and +8 mm, "current_rise": motor j's current
    at +8 mm minus at the start (None if unreadable)}."""
    angles = start.copy()
    # where the tip is now: after the last cable was let out it may not have sprung all the
    # way back (friction with the platform), and that offset must not count as this cable pulling
    ref = vision.measure_tip(6, on_frame) or base
    try:
        current0 = abs(float(cables.read_currents()[j]))
    except Exception:
        current0 = None
    out = {"sign": sign, "k": None, "growth_px": 0.0, "current_rise": None}
    try:
        for k in range(1, int(search_mm) + 1):
            angles[j] = start[j] + sign * k * SEARCH_STEP_RAD
            cables.move_angles(angles)
            m = vision.measure_tip(3, on_frame)
            if m is None:
                raise RuntimeError("Lost the tip during calibration.")
            if np.linalg.norm(m["px"] - ref["px"]) > TAUT_SHIFT_PX:
                out["k"] = k
                break
        if out["k"] is None:
            turned_mm = abs(cables.read_angles()[j] - start[j]) * PULLEY_RADIUS_MM   # did the motor follow?
            commanded_mm = int(search_mm) * SEARCH_STEP_RAD * PULLEY_RADIUS_MM
            log(f"Motor {j}: no tip movement in direction {sign:+d} (the motor turned {turned_mm:.0f} of "
                f"{commanded_mm:.0f} mm commanded)")
            if turned_mm < 0.5 * commanded_mm:
                raise RuntimeError(f"Motor {j} did not follow its commands: check its power and wiring.")
            return out
        seen = []
        for extra in (CONFIRM_PULL_MM, 2 * CONFIRM_PULL_MM):
            angles[j] = start[j] + sign * (out["k"] + extra) * SEARCH_STEP_RAD
            cables.move_angles(angles)
            seen.append(vision.measure_tip(3, on_frame))
        if seen[0] is not None and seen[1] is not None:
            out["growth_px"] = float(np.linalg.norm(seen[1]["px"] - seen[0]["px"]))
        if current0 is not None:
            try:
                out["current_rise"] = abs(float(cables.read_currents()[j])) - current0
                if not np.isfinite(out["current_rise"]):
                    out["current_rise"] = None
            except Exception:
                out["current_rise"] = None
        return out
    finally:
        cables.move_angles(start)
        vision.forget_tip(expect=base["pos"])


def reachability(calib, tip_rest, tol_mm=6.0):
    """reachable((x, z)) -> bool from the calibrated model: can the cables (pull-only, up to their
    limit) bring the tip there from the rest pose? Used to keep detour waypoints reachable."""
    J = np.array(calib["J"])[[0, 2], :]
    rest = np.asarray(tip_rest, float)[[0, 2]]
    p_max = calib["pull_max_mm"]
    return lambda point: solve_pulls(J, np.asarray(point, float) - rest, p_max)[1] <= tol_mm


def move_tip_along(cables, vision, waypoints, target, on_frame, tol_mm=4.0, waypoint_tol_mm=8.0,
                   adapt=False, hold_y=None, y_weight=0.15, watch=None, log=print):
    """Follow a planned path: each (x, z) waypoint with a loose tolerance, then `target` (robot
    frame) with tol_mm. adapt=True learns the cable model as the arm moves (needed in contact).
    hold_y (adaptive only): also try to keep the tip at this height along the way (raised over the
    letters). It is only a weak extra goal (y_weight): 1 mm of sideways error counts as much as
    1/y_weight mm of height, so it never pulls the tip off the way to the letter.
    watch (adaptive only): see move_tip_adaptive; a "stuck" waypoint ends the whole route.
    Returns move_tip_to's result for the final target, plus "waypoints"."""
    quiet = lambda *a, **k: None
    extra = {"hold_y": hold_y, "y_weight": y_weight} if adapt and hold_y is not None else {}
    if adapt and watch is not None:
        extra["watch"] = watch
    move = move_tip_adaptive if adapt else move_tip_to
    for x, z in waypoints[:-1]:
        step = move(cables, vision, np.array([x, target[1], z]), on_frame,
                    tol_mm=waypoint_tol_mm, max_iter=8, log=quiet, **extra)
        if step["reason"] in ("lost the tip", "stuck"):
            return {**step, "waypoints": len(waypoints)}
    return {**move(cables, vision, target, on_frame, tol_mm=tol_mm, log=log, **extra), "waypoints": len(waypoints)}


def move_tip_adaptive(cables, vision, target, on_frame, tol_mm=4.0, gain=0.7, max_iter=30,
                      max_step_mm=3.0, learn_rate=0.5, hold_y=None, y_weight=1.0, y_tol_mm=None, watch=None,
                      log=print):
    """Closed loop with a cable model that is corrected after every move (Broyden update).
    When the arm lies against an obstacle, the obstacle changes how each cable moves the tip, so
    the calibrated model is wrong; each measured move fixes the model in the direction just tried:
        J += learn_rate * (actual_move - J @ pull_change) pull_change^T / |pull_change|^2
    The learned 3x4 model (x, y, z) is kept in cables.live_J (reset with reset_live_model).
    hold_y: also steer the tip's height toward hold_y (robot y, mm), weighted by y_weight against
    the sideways error, e.g. to keep it raised while it travels. Reached = sideways error < tol_mm
    (and, with y_tol_mm, height within y_tol_mm of hold_y). watch(tip) -> True stops the move with
    reason "stuck" (called after every measured step, e.g. a Watchdog). Same result dict as move_tip_to."""
    if getattr(cables, "live_J", None) is None:
        reset_live_model(cables)
    J = cables.live_J
    p_max = np.array(cables.calib["pull_max_mm"])
    w = np.array([1.0, y_weight if hold_y is not None else 0.0, 1.0])     # weights of the x, y, z errors

    def result(reached, error, steps, tip, reason):
        return {"reached": reached, "error_mm": error, "steps": steps,
                "tip": None if tip is None else tip["pos"], "reason": reason}

    def errors(tip):
        x, y, z = tip["pos"]
        err = np.array([target[0] - x, (hold_y - y) if hold_y is not None else 0.0, target[2] - z])
        return err, float(np.hypot(err[0], err[2]))

    def done(err, e):
        return e < tol_mm and (y_tol_mm is None or abs(err[1]) < y_tol_mm)

    tip = vision.measure_tip(5, on_frame)
    if tip is None:
        return result(False, math.nan, 0, None, "tip not visible")
    best, stalled = math.inf, 0
    for it in range(1, max_iter + 1):
        err, e = errors(tip)
        if done(err, e):
            return result(True, e, it - 1, tip, "reached" if it > 1 else "already there")
        # pulls whose predicted move is gain * error (cables only pull, each within its limit)
        before = cables.pulls.copy()
        p_new, _ = solve_pulls(w[:, None] * J, w * (J @ before + gain * err), p_max)
        change = np.clip(p_new - before, -max_step_mm, max_step_mm)
        if np.linalg.norm(change) < 0.2:
            return result(False, e, it - 1, tip, "out of reach")    # the model sees no way to get closer
        cables.set_pulls(before + change)
        new_tip = vision.measure_tip(3, on_frame)
        if new_tip is None:
            return result(False, e, it, None, "lost the tip")
        dp = cables.pulls - before
        moved = new_tip["pos"] - tip["pos"]
        predicted = J @ dp
        # learn from moves of >= 0.5 mm of cable; but a tip that should have moved a lot and barely
        # did is caught (friction, a block's edge): that says nothing about the cables, so don't
        # learn "this cable does nothing" from it
        stuck = np.linalg.norm(moved[[0, 2]]) < 1.0 and np.linalg.norm(predicted[[0, 2]]) > 3.0
        if np.dot(dp, dp) > 0.25 and not stuck:
            J += learn_rate * np.outer(moved - predicted, dp) / np.dot(dp, dp)
            _keep_columns(J, cables)
        tip = new_tip
        err, e_new = errors(tip)
        log(f"  step {it:2d}: error {e_new:5.1f} mm | pulls {np.round(cables.pulls, 1)} mm | "
            f"tip {tip['pos'][1] - target[1]:+.0f} mm above target")
        # caught on something (or out of time); not while it is already close: tiny final steps
        # are normal there, and the stall check below ends those
        if watch is not None and watch(tip, cables.pulls) and (watch.timed_out or e_new > 1.5 * tol_mm):
            return result(False, e_new, it, tip, "stuck")
        progress = float(np.linalg.norm(w * err))
        if progress < best - 0.5:
            best, stalled = progress, 0
        else:
            stalled += 1
            if stalled >= 5:
                return result(False, e_new, it, tip, "out of reach")
    err, e = errors(tip)
    return result(done(err, e), e, max_iter, tip, "reached" if done(err, e) else "step limit")


class Watchdog:
    """Tells a move to give up: the tip has stayed within stuck_mm (sideways) for stuck_s seconds
    although the cables kept changing (at least stuck_cable_mm in that time: it's caught on
    something, not just waiting), or the whole attempt took longer than timeout_s.
    Call it after every measured move: watch(tip, pulls) -> True = give up (reason in .why)."""

    def __init__(self, stuck_s=3.0, stuck_mm=4.0, timeout_s=60.0, stuck_cable_mm=8.0):
        self.stuck_s, self.stuck_mm, self.timeout_s = stuck_s, stuck_mm, timeout_s
        self.stuck_cable_mm = stuck_cable_mm
        self.start, self.history, self.why, self.timed_out = time.time(), [], None, False

    def __call__(self, tip, pulls=None):
        now = time.time()
        self.history.append((now, tip["pos"][[0, 2]].copy(), None if pulls is None else np.array(pulls, float)))
        if now - self.start > self.timeout_s:
            self.why, self.timed_out = f"took longer than {self.timeout_s:.0f} s", True
            return True
        window = [h for h in self.history if now - h[0] <= self.stuck_s]
        covered = now - window[0][0]                 # how much of the last stuck_s seconds we have seen
        spread = max(np.linalg.norm(p - window[-1][1]) for _, p, _ in window)
        cable = [q for _, _, q in window if q is not None]
        driven = sum(np.abs(b - a).sum() for a, b in zip(cable, cable[1:])) if len(cable) == len(window) else math.inf
        if (covered >= 0.8 * self.stuck_s and len(window) >= 3 and spread < self.stuck_mm
                and driven >= self.stuck_cable_mm):
            self.why = (f"tip stayed within {self.stuck_mm:.0f} mm for {self.stuck_s:.0f} s"
                        + (f" while the cables moved {driven:.0f} mm" if driven != math.inf else ""))
            return True
        return False


def reset_live_model(cables):
    """Start the adaptive model from the calibration (e.g. back at the rest pose)."""
    set_live_model(cables, cables.calib["J"])


def set_live_model(cables, J):
    """Use a freshly measured model (probe_model) from here on."""
    cables.live_J = np.array(J, float).copy()
    cables.model_floor = np.linalg.norm(cables.live_J[[0, 2]], axis=0)


def _keep_columns(J, cables, floor=0.3):
    """Don't let learning shrink a cable's sideways effect below `floor` of what it was when the
    model was set: a few moves in contact must not make the controller give up on a cable."""
    if getattr(cables, "model_floor", None) is None:
        cables.model_floor = np.linalg.norm(J[[0, 2]], axis=0).copy()
    norms = np.linalg.norm(J[[0, 2]], axis=0)
    low = norms < floor * cables.model_floor
    for j in np.flatnonzero(low & (norms > 1e-9)):
        J[:, j] *= floor * cables.model_floor[j] / norms[j]


def _cable_load(cables, baseline):
    """Largest rise in motor current (mA) since `baseline` over the used motors; 0 if unreadable."""
    try:
        rise = np.abs(cables.read_currents()) - baseline
    except Exception:
        return 0.0
    rise = rise[(cables.used > 0) & np.isfinite(rise)]
    return float(rise.max()) if len(rise) else 0.0


def probe_model(cables, vision, on_frame, probe_mm=RAISE_PROBE_MM):
    """Measure the local 3x4 cable model where the arm is now: pull each used cable probe_mm (or
    release it, if it's at its limit), measure the tip, put the cable back.
    Returns (J, tip at the current pulls), or (None, tip) if the tip was lost."""
    p0, p_max = cables.pulls.copy(), np.array(cables.calib["pull_max_mm"])
    base = vision.measure_tip(5, on_frame)
    if base is None:
        return None, None
    J = np.zeros((3, 4))
    for j in np.flatnonzero(cables.used):
        for size in (probe_mm, 2 * probe_mm):                 # barely moved (friction)? try a bigger pull
            d = size if p0[j] + size <= p_max[j] else -min(size, p0[j])
            if abs(d) < 0.5:
                break
            p = p0.copy()
            p[j] += d
            cables.set_pulls(p)
            m = vision.measure_tip(3, on_frame)
            cables.set_pulls(p0)
            if m is None:
                return None, base
            J[:, j] = (m["pos"] - base["pos"]) / d
            base = vision.measure_tip(3, on_frame) or base    # the tip doesn't always come back exactly
            if np.linalg.norm(J[:, j]) * abs(d) >= 2.0:       # moved at least 2 mm: measured well enough
                break
    return J, base


def describe_model(J, used):
    """One line per the model: where 1 mm of each cable moves the tip (sideways direction, up/down)."""
    parts = []
    for j in np.flatnonzero(used):
        sideways = J[[0, 2], j]
        parts.append(f"M{j} {np.linalg.norm(sideways):.1f} mm/mm at {math.degrees(math.atan2(sideways[1], sideways[0])):+.0f} deg, "
                     f"{J[1, j]:+.1f} up")
    return " | ".join(parts)


def _heading(J, j, sign, to):
    """cos of the angle between where cable j moves the tip sideways (pulled: sign +1, let out: -1)
    and the direction `to`; 1 if there's no direction to keep to."""
    if to is None:
        return 1.0
    v = sign * J[[0, 2], j]
    return float(np.dot(v, to) / (np.linalg.norm(v) * np.linalg.norm(to) + 1e-9))


def raise_tip(cables, vision, on_frame, J=None, step_mm=RAISE_STEP_MM, max_iter=12,
              max_load_ma=RAISE_MAX_LOAD_MA, stop_at_y=None, toward_xz=None, min_heading_cos=0.5,
              near_mm=15.0, log=print):
    """Bend the arm to raise the tip before it moves over the letters (so it doesn't drag them
    or run into the side of a raised one).
    Tendons can't shorten the arm (pulling all cables together only stiffens it): the tip rises
    only by bending, which also moves it sideways. So this climbs the height measured by the camera,
    one cable at a time: pull the cable that raises the tip most (at rest every cable raises it a
    little, but pulling them together just stiffens the arm), let out cables that hold it down,
    correct the model from each measured move (Broyden), and measure it again (probe_model) when a
    step doesn't raise the tip. J: the local model, if just measured (else it is measured first).
    toward_xz (the next letter): only use cables whose sideways push points within ~60 deg
    (min_heading_cos) of the way to it, so raising never takes the tip away from the letter; stops
    once the tip is within near_mm of it sideways. stop_at_y: stop once the tip is this high.
    Also stops when no allowed cable raises the tip any further, a motor's current rises by
    max_load_ma (cables getting tight), or the tip is lost; ends at the highest pose found.
    Leaves the model in cables.live_J for the moves that follow.
    Returns {"raised_mm", "tip", "reason", "steps"}."""
    if J is None:
        J, tip = probe_model(cables, vision, on_frame)
    else:
        tip = vision.measure_tip(5, on_frame)
    if tip is None or J is None:
        return {"raised_mm": 0.0, "tip": tip, "reason": "lost the tip", "steps": 0}
    J = J.copy()
    goal = None if toward_xz is None else np.asarray(toward_xz, float)
    p_max, used = np.array(cables.calib["pull_max_mm"]), cables.used > 0
    start_y = best_y = float(tip["pos"][1])
    best_pulls, best_tip = cables.pulls.copy(), tip
    try:
        baseline = np.abs(cables.read_currents())
    except Exception:
        baseline = np.zeros(4)
    if stop_at_y is not None and start_y >= stop_at_y:
        set_live_model(cables, J)
        return {"raised_mm": 0.0, "tip": tip, "reason": "already high enough", "steps": 0}
    no_way = "no cable raises the tip" + (" while moving it toward the letter" if goal is not None else " any further")
    fresh, reason, it = True, no_way, 0                   # fresh = J was just measured
    for it in range(1, max_iter + 1):
        p = cables.pulls.copy()
        to = None if goal is None else goal - tip["pos"][[0, 2]]
        if to is not None and np.linalg.norm(to) < near_mm:
            reason = "over the letter"
            break
        g = np.where(used, J[1], 0.0)                     # mm of height per mm of each cable, here
        up = [j for j in np.flatnonzero(g > 0)
              if p[j] < p_max[j] - 0.1 and step_mm * g[j] >= 0.5 and _heading(J, j, +1, to) >= min_heading_cos]
        down = [j for j in np.flatnonzero(g < 0)          # cables holding the tip down: let them out
                if p[j] > 0.1 and min(step_mm, p[j]) * -g[j] >= 0.5 and _heading(J, j, -1, to) >= min_heading_cos]
        if not up and not down:
            if fresh:
                break
            J, _ = probe_model(cables, vision, on_frame)
            fresh = True
            if J is None:
                reason = "lost the tip"
                break
            continue
        g_max = max(abs(g[j]) for j in up + down)
        dp = np.zeros(4)
        for j in down:
            dp[j] = -min(step_mm * -g[j] / g_max, p[j])
        if up:                                            # pull the one cable that raises it most
            j = max(up, key=lambda j: g[j] * (1.5 + _heading(J, j, +1, to)))
            dp[j] = step_mm
        cables.set_pulls(p + dp)
        load = _cable_load(cables, baseline)
        new_tip = vision.measure_tip(3, on_frame)
        if new_tip is None:
            reason = "lost the tip"
            break
        dp = cables.pulls - p
        if np.dot(dp, dp) > 0.25:
            J += 0.5 * np.outer(new_tip["pos"] - tip["pos"] - J @ dp, dp) / np.dot(dp, dp)
        tip = new_tip
        y = float(tip["pos"][1])
        log(f"  raise step {it:2d}: tip {y - start_y:+5.1f} mm higher | pulls {np.round(cables.pulls, 1)} mm | "
            f"motor load +{load:.0f} mA")
        if load > max_load_ma:
            reason = f"cables getting tight (motor current +{load:.0f} mA)"
            break
        if stop_at_y is not None and y >= stop_at_y:
            best_y, best_pulls, best_tip, reason = y, cables.pulls.copy(), tip, "high enough"
            break
        if y > best_y + RAISE_MIN_GAIN_MM:
            best_y, best_pulls, best_tip, fresh = y, cables.pulls.copy(), tip, False
        elif fresh:
            reason = "tip stopped rising"                 # even with a freshly measured model
            break
        else:
            J, _ = probe_model(cables, vision, on_frame)  # the model may be off: measure it again
            fresh = True
            if J is None:
                reason = "lost the tip"
                break
    else:
        reason = "step limit"
    if not np.allclose(cables.pulls, best_pulls):
        cables.set_pulls(best_pulls)                      # back to the highest pose found
        best_tip = vision.measure_tip(5, on_frame) or best_tip
    if J is not None:
        set_live_model(cables, J)
    return {"raised_mm": float(best_tip["pos"][1]) - start_y, "tip": best_tip, "reason": reason, "steps": it}


def lower_onto(cables, vision, target, on_frame, tol_mm=8.0, step_mm=4.0, log=print):
    """Come down onto the letter: lower the height goal a few mm at a time while steering the tip
    over target (x, z), until it rests on it (stops going down) or reaches the letter's height,
    then correct x, z. Returns move_tip_adaptive's result plus "lowered_mm" and "settle"."""
    quiet = lambda *a, **k: None
    tip = vision.measure_tip(5, on_frame)
    if tip is None:
        return {"reached": False, "error_mm": math.nan, "steps": 0, "tip": None, "reason": "tip not visible",
                "lowered_mm": 0.0, "settle": "tip not visible"}
    start_y = lowest = y = float(tip["pos"][1])
    no_drop, settle, steps = 0, "at the letter's height", 0
    for _ in range(int(max(0.0, y - target[1]) / step_mm) + 4):
        if y <= target[1] + 1.0:
            break
        r = move_tip_adaptive(cables, vision, target, on_frame, tol_mm=tol_mm, max_iter=4,
                              hold_y=max(target[1], y - step_mm), y_weight=0.5, y_tol_mm=1.5, log=quiet)
        steps += r["steps"]
        if r["tip"] is None:
            settle = "lost the tip"
            break
        y = float(r["tip"][1])
        log(f"  lowering: tip {y - start_y:+5.1f} mm, {r['error_mm']:4.1f} mm from the letter sideways")
        if y < lowest - RAISE_MIN_GAIN_MM:
            lowest, no_drop = y, 0
        else:
            no_drop += 1
            if no_drop >= 2:
                settle = "resting on it"
                break
    else:
        settle = "step limit"
    final = move_tip_adaptive(cables, vision, target, on_frame, tol_mm=tol_mm, max_iter=6, log=quiet)
    end_y = final["tip"][1] if final["tip"] is not None else y
    return {**final, "steps": steps + final["steps"], "lowered_mm": start_y - float(end_y), "settle": settle}


def move_tip_to(cables, vision, target, on_frame, tol_mm=4.0, gain=0.6, max_iter=25,
                max_step_mm=4.0, log=print):
    """Closed loop: bring the tip's horizontal position (x, z) over `target` (robot frame, mm).
    The vertical position can't be set independently: bending the arm also lifts the tip.
    Returns {"reached", "error_mm", "steps", "tip", "reason"}."""
    J = np.array(cables.calib["J"])[[0, 2], :]
    p_max = np.array(cables.calib["pull_max_mm"])
    goal = np.array([target[0], target[2]], float)

    def result(reached, error, steps, tip, reason):
        return {"reached": reached, "error_mm": error, "steps": steps,
                "tip": None if tip is None else tip["pos"], "reason": reason}

    tip = vision.measure_tip(5, on_frame)
    if tip is None:
        log("Tip not visible.")
        return result(False, math.nan, 0, None, "tip not visible")
    e = float(np.linalg.norm(goal - tip["pos"][[0, 2]]))
    if e < tol_mm:
        return result(True, e, 0, tip, "already there")
    offset = goal - tip["pos"][[0, 2]] + J @ cables.pulls     # displacement from the all-taut pose
    correction = np.zeros(2)
    best_err, stalled, warned = math.inf, 0, False

    for it in range(1, max_iter + 1):
        p_goal, residual = solve_pulls(J, offset + correction, p_max)
        rate_limited = np.any(np.abs(p_goal - cables.pulls) > max_step_mm)
        cables.set_pulls(cables.pulls + np.clip(p_goal - cables.pulls, -max_step_mm, max_step_mm))

        tip = vision.measure_tip(3, on_frame)
        if tip is None:
            log("Lost the tip; stopping.")
            return result(False, e, it, None, "lost the tip")
        err = goal - tip["pos"][[0, 2]]
        e = float(np.linalg.norm(err))
        log(f"  step {it:2d}: horizontal error {e:5.1f} mm | pulls {np.round(cables.pulls, 1)} mm | "
            f"tip {tip['pos'][1] - target[1]:+.0f} mm above target")
        if e < tol_mm:
            return result(True, e, it, tip, "reached")
        if rate_limited:
            continue                     # still travelling toward the planned pulls
        if residual > tol_mm and np.any(cables.pulls >= p_max - 0.1) and not warned:
            log("  Planned pulls hit the cable limit: the target may be outside the reachable area.")
            warned = True
        correction = correction + gain * err
        if np.linalg.norm(correction) > 60:
            correction *= 60 / np.linalg.norm(correction)
        if e < best_err - 0.5:
            best_err, stalled = e, 0
        else:
            stalled += 1
            if stalled >= 4:
                log("  No further progress: the target is probably out of reach from here.")
                return result(False, e, it, tip, "out of reach")
    return result(False, e, max_iter, tip, "step limit")
