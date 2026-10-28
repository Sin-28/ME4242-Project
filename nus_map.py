"""
Measured workspace map: where the tip actually goes for combinations of cable pulls, with the
obstacles in place.

The calibration's linear model (tip = tip_taut + J p) only holds near the rest pose: the real arm
bends nonlinearly, a tip on the platform sticks and slips, and around the obstacles it leans on
them or gets caught. So, once per calibration (`python nus_main.py explore`, obstacles and holder
in place, letters removed: the magnet would pick them up), every pair of neighbouring cables is
stepped through a grid of pulls, as far as the arm goes, and the tip's position is recorded.

The samples form a graph: each is linked to its neighbours on the grid (one cable 1 step more or
less), so a route is a chain of pull settings the arm has already been in, through tip positions
it was actually seen at, contacts with the obstacles included. Playing such a chain back
reproduces what was observed. Also:
  * reachable(point): the tip was seen near there;
  * pulls_for(point): pulls that put the tip there (local fit to the nearest samples), for short moves.
"""

import heapq
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from nus_path import convex_hull, densify, inside

HERE = Path(__file__).resolve().parent
MAP_PATH = HERE / "workspace_map.json"


# ---------------------------------------------------------------- building it

def neighbour_pairs(calib):
    """Used cables in order of the direction they pull the tip, and each neighbouring pair:
    two cables either side of a direction reach every point between their directions. Cables
    pulling about opposite ways (more than 135 deg apart, e.g. with a cable missing) aren't paired:
    pulling both only stiffens the arm."""
    J = np.array(calib["J"], float)
    used = [j for j in range(4) if calib["pull_sign"][j] != 0 and np.linalg.norm(J[[0, 2], j]) > 1e-6]
    order = sorted(used, key=lambda j: math.atan2(J[2, j], J[0, j]))
    if len(order) < 2:
        return []
    pairs = [tuple(order)] if len(order) == 2 else [(order[i], order[(i + 1) % len(order)]) for i in range(len(order))]
    angle = {j: math.degrees(math.atan2(J[2, j], J[0, j])) for j in order}
    return [(a, b) for a, b in pairs if abs((angle[a] - angle[b] + 180.0) % 360.0 - 180.0) <= 135.0]


def explore(cables, vision, on_frame, pull_max_mm, step_mm=5.0, max_load_ma=1500.0, max_lag_mm=3.0,
            still_mm=1.0, still_steps=3, only_motors=None, push=(), log=print):
    """For each neighbouring cable pair (a, b): rows of increasing pull on a, sweeping b up and
    down (zigzag, so every move is one step), recording the tip after each move. While a row pulls
    harder it stops (and the next rows stay below that) when:
      * the tip stopped moving: still_steps moves in a row of less than still_mm (more cable
        would only add tension: that's as far as the arm bends there),
      * a motor can't reach its position (max_lag_mm behind: pulling as hard as it can),
      * a motor's current rose by max_load_ma (safety), or the tip is lost,
      * pull_max_mm is reached.
    Between pairs the cables are let out step by step. only_motors: explore only the pairs with one
    of these motors and keep the rest of the saved map (same calibration). push: motors whose rows
    don't stop when the tip stops moving (pressed on an obstacle: pulling harder gets it past),
    only on the current / lag / pull limits. Saves and returns the map."""
    from nus_control import _cable_load
    pairs = neighbour_pairs(cables.calib)
    if not pairs:
        raise RuntimeError("Need at least 2 working cables in the calibration to explore.")
    keep = []                                         # samples of the pairs not explored again
    if only_motors:
        old = json.loads(MAP_PATH.read_text()) if MAP_PATH.exists() else None
        if old is None or old.get("calibration") != cables.calib.get("created") or old.get("pairs") != [list(p) for p in pairs]:
            raise RuntimeError("--only needs a saved map from this calibration: run a full explore first.")
        redo = {k for k, p in enumerate(pairs) if set(p) & set(only_motors)}
        keep = [row for row in old["samples"] if int(row[7]) >= 0 and int(row[7]) not in redo]
        log(f"Exploring again only {', '.join(f'M{a}+M{b}' for k, (a, b) in enumerate(pairs) if k in redo)}; "
            f"keeping {len(keep)} positions of the other pairs")
    levels = np.arange(0.0, pull_max_mm + 1e-6, step_mm)
    cables.set_pulls(np.zeros(4))
    vision.forget_tip()
    rest = vision.measure_tip(8, on_frame)
    if rest is None:
        raise RuntimeError("Can't see the green tip at the rest pose.")
    samples = [{"pulls": np.zeros(4), "tip": rest["pos"], "pair": -1, "i": 0, "j": 0}]
    try:
        baseline = np.abs(cables.read_currents())
    except Exception:
        baseline = np.zeros(4)
    check_cables(cables, vision, on_frame, rest, push=push, log=log)

    def too_hard(moved):
        load = _cable_load(cables, baseline)
        lag = float(max(cables.lag_mm()[m] for m in moved))
        if load > max_load_ma:
            return f"motor current +{load:.0f} mA"
        if lag > max_lag_mm:
            return f"motor {lag:.1f} mm short of its position"
        return None

    for k, (a, b) in enumerate(pairs):
        if only_motors and not set((a, b)) & set(only_motors):
            continue
        top = len(levels) - 1                         # highest j (pull on b) the rows may still go to
        reached = 0.0
        pushing = bool(set((a, b)) & set(push))
        for i, pa in enumerate(levels):
            up = i % 2 == 0
            js = range(0, top + 1) if up else range(top, -1, -1)
            still, prev, row_ok, unseen, last_seen_j = 0, None, False, 0, 0
            for j in js:
                p = np.zeros(4)
                p[a], p[b] = pa, levels[j]
                cables.set_pulls(p)
                why = too_hard((a, b))
                if why is not None:
                    if up or j == 0:
                        log(f"  M{a} {pa:.0f} + M{b} {levels[j]:.0f} mm: {why}, not pulling further here")
                        top = max(j - 1, 0) if up else top
                        break
                    continue                                  # letting out: just go on
                m = vision.measure_tip(3, on_frame)
                if m is None:                                 # hidden (behind the arm, an obstacle): skip it
                    unseen += 1
                    if up and unseen >= 3:                    # out of sight for a while: don't pull further blind
                        log(f"  M{a} {pa:.0f} + M{b} {levels[j]:.0f} mm: tip out of sight, not pulling further here")
                        top = last_seen_j
                        break
                    continue
                unseen, last_seen_j = 0, j
                samples.append({"pulls": cables.pulls.copy(), "tip": m["pos"], "pair": k, "i": i, "j": j})
                row_ok = True
                reached = max(reached, pa, levels[j])
                if up and prev is not None and not pushing:
                    still = still + 1 if np.linalg.norm(m["pos"][[0, 2]] - prev) < still_mm else 0
                    if still >= still_steps:
                        log(f"  M{a} {pa:.0f} + M{b} {levels[j]:.0f} mm: the tip stopped moving, not pulling further here")
                        top = j
                        break
                prev = m["pos"][[0, 2]]
            if not row_ok and i > 0:
                log(f"  M{a} at {pa:.0f} mm: nothing more to record, next pair")
                break
            # rows start with cable b let out: if the tip hasn't moved for a few rows as a got
            # pulled harder, a is as far as it goes here
            first = [smp for smp in samples if smp["pair"] == k and smp["j"] == 0]
            if not (a in push) and len(first) > still_steps and all(
                    np.linalg.norm(first[-1 - n]["tip"][[0, 2]] - first[-2 - n]["tip"][[0, 2]]) < still_mm
                    for n in range(still_steps)):
                log(f"  M{a} {pa:.0f} mm: the tip stopped moving as M{a} pulled harder, not pulling further")
                break
        log(f"Cables M{a} + M{b}: pulled up to {reached:.0f} mm")
        for frac in (0.66, 0.33, 0.0):                        # let out gradually: the tip may be on something
            cables.set_pulls(cables.pulls * frac)
        vision.forget_tip(expect=rest["pos"])
    data = {"created": datetime.now().isoformat(timespec="seconds"), "calibration": cables.calib.get("created"),
            "pull_max_mm": float(pull_max_mm), "step_mm": float(step_mm), "rest": rest["pos"].tolist(),
            "pairs": [list(p) for p in pairs],
            "samples": [list(map(float, s["pulls"])) + list(map(float, s["tip"])) + [s["pair"], s["i"], s["j"]]
                        for s in samples] + keep}
    MAP_PATH.write_text(json.dumps(data))
    wmap = WorkspaceMap(data)
    log(f"Saved {MAP_PATH.name}: {len(samples) + len(keep)} tip positions; the tip reaches up to "
        f"{wmap.reach_mm():.0f} mm from the middle")
    return wmap


def check_cables(cables, vision, on_frame, rest, levels_mm=(5.0, 10.0, 15.0, 20.0), min_move_mm=5.0, push=(),
                 log=print):
    """Pull each used cable alone through levels_mm and check the tip keeps moving: at least
    min_move_mm between the first and the last pull at which the camera sees it. A cable that
    doesn't is being let out instead of pulled (wrong direction in the calibration: letting out a
    tight cable moves the tip at first, then nothing), is slack, or is off its spool: exploring
    with it would just unwind it, so stop. If the tip goes out of sight (hidden behind the arm or
    an obstacle) that is only reported: that side gets mapped as far as the camera can see it."""
    bad = []
    for j in np.flatnonzero(cables.used):
        seen = []
        for mm in levels_mm:
            p = np.zeros(4)
            p[j] = mm
            cables.set_pulls(p)
            m = vision.measure_tip(3, on_frame)
            if m is not None:
                seen.append((mm, m["pos"][[0, 2]]))
        for frac in (0.5, 0.0):                      # let out gradually
            cables.set_pulls(cables.pulls * frac)
        vision.forget_tip(expect=rest["pos"])
        hidden = [mm for mm in levels_mm if mm not in [s_[0] for s_ in seen]]
        note = f" (tip not visible at {', '.join(f'{mm:.0f}' for mm in hidden)} mm)" if hidden else ""
        if len(seen) < 2:
            log(f"  M{j} alone: the camera can't see the tip when M{j} pulls{note}. Carrying on; that side is "
                "only mapped where the tip is visible (it may be hidden behind the arm or an obstacle).")
            continue
        first = float(np.linalg.norm(seen[0][1] - rest["pos"][[0, 2]]))
        more = float(np.linalg.norm(seen[-1][1] - seen[0][1]))
        log(f"  M{j} alone: {seen[0][0]:.0f} mm moves the tip {first:.0f} mm, {seen[-1][0]:.0f} mm "
            f"{more:.0f} mm further{note}")
        if more < min_move_mm:
            if j in push:
                log(f"  M{j}: pressed on something; pushing it further while exploring (--push)")
            else:
                bad.append(int(j))
    if bad:
        raise RuntimeError(
            f"Cable(s) {', '.join(f'M{j}' for j in bad)} stop moving the tip when pulled further: their direction "
            "in the calibration is probably wrong (the motor lets the cable out), or the cable is slack / off its "
            "spool. Check with `python nus_main.py jog` which way tightens each cable, then "
            "`python nus_main.py calibrate --pull-signs=...` and explore again.")


def load_map(calib=None, pull_limit=None, log=print):
    """The saved map, or None (missing, or made with a different calibration). pull_limit: leave out
    the samples that need more cable than that."""
    if not MAP_PATH.exists():
        return None
    data = json.loads(MAP_PATH.read_text())
    if calib is not None and data.get("calibration") != calib.get("created"):
        log(f"{MAP_PATH.name} was made with another calibration ({data.get('calibration')}): "
            "run `python nus_main.py explore` again. Not using it.")
        return None
    return WorkspaceMap(data, pull_limit)


# ---------------------------------------------------------------- using it

class WorkspaceMap:
    def __init__(self, data, pull_limit=None):
        s = np.array(data["samples"], float)
        if pull_limit is not None:                  # only what the cables may pull now
            s = s[(s[:, :4] <= pull_limit + 1e-6).all(axis=1)]
        self.pulls, self.tips = s[:, :4], s[:, 4:7]
        self.grid = s[:, 7:10].astype(int) if s.shape[1] >= 10 else None    # (pair, i, j) per sample
        self.xz = self.tips[:, [0, 2]]
        self.rest = np.array(data["rest"], float)
        self.step_mm = float(data.get("step_mm", 5.0))
        self.pull_max = np.maximum(self.pulls.max(axis=0), 0.0)
        # spacing of the samples in tip space, around each sample (they spread out as the arm bends
        # further): a point counts as reachable within about the local spacing of its nearest sample
        d = np.linalg.norm(self.xz[:, None] - self.xz[None], axis=2)
        np.fill_diagonal(d, np.inf)
        k = min(3, len(self.xz) - 1)
        self.local_spacing = np.sort(d, axis=1)[:, :k].mean(axis=1) if k > 0 else np.full(len(self.xz), 10.0)
        self.spacing = float(np.median(self.local_spacing))
        self._hull = convex_hull(self.xz) if len(self.xz) >= 3 else None
        self._links = self._build_links() if self.grid is not None else None

    # -- reach
    def reach_mm(self):
        return float(np.linalg.norm(self.xz - self.rest[[0, 2]], axis=1).max())

    def outline(self):
        """Convex outline of where the tip was seen (for drawing)."""
        return convex_hull(self.xz)

    def distance_to_reach(self, xz):
        """How far (mm) a point is from the nearest place the tip was seen."""
        return float(np.linalg.norm(self.xz - np.asarray(xz, float), axis=1).min())

    def tolerance_at(self, xz):
        """How far from the nearest sample a point may be and still count as reachable there."""
        i = int(np.argmin(np.linalg.norm(self.xz - np.asarray(xz, float), axis=1)))
        return max(8.0, 1.2 * float(self.local_spacing[i]))

    def reachable(self, xz, tol_mm=None):
        d, tol = self.distance_to_reach(xz), (self.tolerance_at(xz) if tol_mm is None else tol_mm)
        # between samples inside the explored area also counts (it was swept on both sides)
        return d <= tol or (self._hull is not None and inside(np.asarray(xz, float), self._hull) and d <= 2 * tol)

    # -- pulls for a point (short moves, final corrections)
    def pulls_for(self, xz, k=6):
        """Pulls that put the tip at xz: a weighted affine fit of pulls against tip position over the
        k nearest samples (falls back to the weighted mean if they are nearly in a line)."""
        xz = np.asarray(xz, float)
        d = np.linalg.norm(self.xz - xz, axis=1)
        near = np.argsort(d)[:k]
        w = 1.0 / (d[near] + 0.5 * self.spacing)
        A = np.column_stack([self.xz[near], np.ones(len(near))]) * w[:, None]
        if np.linalg.matrix_rank(A, tol=1e-6) == 3 and d[near[0]] < 3 * self.spacing:
            coef = np.linalg.lstsq(A, self.pulls[near] * w[:, None], rcond=None)[0]
            p = np.array([*xz, 1.0]) @ coef
            lo, hi = self.pulls[near].min(axis=0), self.pulls[near].max(axis=0)
            p = np.clip(p, lo - self.step_mm, hi + self.step_mm)          # don't extrapolate far
        else:
            p = (self.pulls[near] * w[:, None]).sum(axis=0) / w.sum()
        return np.clip(p, 0.0, self.pull_max)

    # -- routes through the recorded settings
    def _build_links(self, max_jump_mm=30.0, pinned_mm=1.5):
        """Neighbours: same cable pair, one grid step apart; the same pull setting recorded in two
        pairs (only one cable pulled, or none) is one place. Left out: links where the tip jumped
        more than max_jump_mm in one step (it snapped off or onto something: not repeatable).
        Settings where the tip was pinned (it stayed within pinned_mm of two or more neighbours
        although the cables changed: caught on an obstacle) are marked in self.pinned."""
        n = len(self.pulls)
        key = {}
        for idx in range(n):
            key.setdefault(tuple(np.round(self.pulls[idx], 2)), []).append(idx)
        cell = {(int(g[0]), int(g[1]), int(g[2])): idx for idx, g in enumerate(self.grid)}
        links = [set() for _ in range(n)]
        for idx, (pair, i, j) in enumerate(self.grid):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                other = cell.get((int(pair), int(i) + di, int(j) + dj))
                if other is not None:
                    links[idx].add(other)
                    links[other].add(idx)
        for same in key.values():                     # identical settings: share their neighbours
            together = set().union(*(links[idx] for idx in same)) | set(same)
            for idx in same:
                links[idx] = together - {idx}
        jump = lambda a, b: float(np.linalg.norm(self.xz[a] - self.xz[b]))
        self.snaps = sum(1 for idx in range(n) for nb in links[idx] if nb > idx and jump(idx, nb) > max_jump_mm)
        links = [{nb for nb in l if jump(idx, nb) <= max_jump_mm or np.allclose(self.pulls[idx], self.pulls[nb])}
                 for idx, l in enumerate(links)]
        self.pinned = np.array([sum(1 for nb in links[idx] if jump(idx, nb) < pinned_mm
                                    and not np.allclose(self.pulls[idx], self.pulls[nb])) >= 2
                                for idx in range(n)])
        return [sorted(l) for l in links]

    def has_routes(self):
        return self._links is not None

    def nearest_setting(self, pulls):
        return int(np.argmin(np.abs(self.pulls - np.asarray(pulls, float)).sum(axis=1)))

    def route(self, start_pulls, goal_xz, blocked=(), penalties=(), tol_mm=None, forbid=None, goal_y=None,
              free_near_start_mm=25.0, crossing=None, forbid_cost=None):
        """Cheapest chain of recorded settings from the one nearest start_pulls to one whose tip
        is nearest goal_xz. blocked: [(centre (x, z), radius)] the tip must stay out of (other
        letters); penalties: [(centre, radius, cost)] places to avoid if possible (where an earlier
        try got stuck). forbid: True per setting the route may not pass through (e.g. the letter
        being carried would hit an obstacle there), except within free_near_start_mm of the start.
        goal_y: the tip should get down to this height at the goal (to touch a magnet): among the
        settings near the goal, the one where the tip was lowest wins (height above goal_y counts
        like distance). crossing(a, b) -> True: the move between settings a and b isn't allowed
        (e.g. the carried letter would sweep through an obstacle on the way), except near the start.
        forbid_cost: instead of ruling those out, each one passed costs this much (the least bad way
        when there is no clean one). Cost = tip travel (mm) + a little cable change. Returns
        {"settings": [indices], "pulls": [...], "path": [tip (x, z)], "length", "cost", "end_gap",
        "end_height": tip height above goal_y there (or None), "forbidden": how many forbidden
        settings or moves it passes} or None if the goal can't be reached."""
        goal = np.asarray(goal_xz, float)
        start = self.nearest_setting(start_pulls)
        n = len(self.pulls)

        def is_blocked(idx):
            return any(np.linalg.norm(self.xz[idx] - np.asarray(c, float)) < r for c, r in blocked)

        extra = np.where(getattr(self, "pinned", np.zeros(n, bool)), 60.0, 0.0)   # caught there while exploring
        for c, r, cost in penalties:
            extra[np.linalg.norm(self.xz - np.asarray(c, float), axis=1) < r] += cost
        gap = np.linalg.norm(self.xz - goal, axis=1)
        tol = self.tolerance_at(goal) if tol_mm is None else tol_mm
        def near_start(xz):
            return np.linalg.norm(xz - self.xz[start]) <= free_near_start_mm

        def is_forbidden(idx, nb):
            if forbid is not None and forbid[nb] and gap[nb] > tol and not near_start(self.xz[nb]):
                return True
            return bool(crossing is not None and not near_start(0.5 * (self.xz[idx] + self.xz[nb]))
                        and crossing(idx, nb))

        best, prev, done = {start: 0.0}, {}, set()
        queue = [(0.0, start)]
        while queue:
            cost, idx = heapq.heappop(queue)
            if idx in done:
                continue
            done.add(idx)
            for nb in self._links[idx]:
                if nb in done or (is_blocked(nb) and gap[nb] > tol):
                    continue
                bad = is_forbidden(idx, nb)
                if bad and forbid_cost is None:
                    continue
                step = float(np.linalg.norm(self.xz[nb] - self.xz[idx])) + \
                    0.2 * float(np.abs(self.pulls[nb] - self.pulls[idx]).sum()) + extra[nb] + \
                    (forbid_cost if bad else 0.0)
                if cost + step < best.get(nb, math.inf):
                    best[nb], prev[nb] = cost + step, idx
                    heapq.heappush(queue, (cost + step, nb))
        candidates = [idx for idx in done if gap[idx] <= max(tol, gap[list(done)].min() + 1e-9)]
        if not candidates:
            return None
        # nearest first; among those about as near (within 5 mm), the lowest tip, then the cheapest
        nearest = min(gap[idx] for idx in candidates)
        close = [idx for idx in candidates if gap[idx] <= max(nearest + 5.0, 4.0)]
        high = (lambda idx: max(0.0, self.tips[idx, 1] - goal_y)) if goal_y is not None else (lambda idx: 0.0)
        end = min(close, key=lambda idx: (round(high(idx) / 3.0), best[idx]))
        chain = [end]
        while chain[-1] != start:
            chain.append(prev[chain[-1]])
        chain.reverse()
        path = [self.xz[idx].copy() for idx in chain]
        length = float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))
        return {"settings": chain, "pulls": [self.pulls[idx].copy() for idx in chain], "path": path,
                "length": length, "cost": float(best[end]), "end_gap": float(gap[end]),
                "end_height": None if goal_y is None else float(self.tips[end, 1] - goal_y),
                "forbidden": sum(is_forbidden(a, b) for a, b in zip(chain, chain[1:]))}


# ---------------------------------------------------------------- moving with it

def play(cables, vision, wmap, pull_sequence, target, on_frame, tol_mm=8.0, step_mm=2.5, gain=0.7,
         max_corrections=12, watch=None, log=print):
    """Move the cables through pull_sequence a little at a time (no motor changes more than step_mm
    per move), with the camera measuring after every move, then correct the leftover error at the
    goal by asking the map for the goal shifted by the accumulated error (absorbs friction,
    hysteresis and map errors). watch(tip, pulls) -> True stops with reason "stuck".
    Same result dict as move_tip_to."""
    goal = np.asarray(target, float)[[0, 2]]
    steps = 0

    def result(reached, error, tip, reason):
        return {"reached": reached, "error_mm": error, "steps": steps,
                "tip": None if tip is None else tip["pos"], "reason": reason}

    def gap(tip):
        return math.nan if tip is None else float(np.linalg.norm(tip["pos"][[0, 2]] - goal))

    def go_to(p_goal):
        nonlocal steps
        p_goal, tip = np.asarray(p_goal, float), None
        while True:
            delta = p_goal - cables.pulls
            if np.abs(delta).max() < 0.05:
                return tip, None
            cables.set_pulls(cables.pulls + np.clip(delta, -step_mm, step_mm))
            steps += 1
            tip = vision.measure_tip(3, on_frame)
            if tip is None:
                return None, "lost the tip"
            if watch is not None and watch(tip, cables.pulls) and (watch.timed_out or gap(tip) > 1.5 * tol_mm):
                return tip, "stuck"

    tip = None
    for p in pull_sequence:
        tip, why = go_to(p)
        if why is not None:
            return result(False, gap(tip), tip, why)
    correction, best, stalled = np.zeros(2), math.inf, 0
    for _ in range(max_corrections):
        tip = vision.measure_tip(3, on_frame)
        if tip is None:
            return result(False, math.nan, None, "lost the tip")
        err = goal - tip["pos"][[0, 2]]
        e = float(np.linalg.norm(err))
        log(f"  at the goal: {e:5.1f} mm off | pulls {np.round(cables.pulls, 1)} mm")
        if e < tol_mm:
            return result(True, e, tip, "reached")
        if e < best - 0.5:
            best, stalled = e, 0
        else:
            stalled += 1
            if stalled >= 4:
                break
        correction = correction + gain * err
        if np.linalg.norm(correction) > 40.0:
            correction *= 40.0 / np.linalg.norm(correction)
        tip, why = go_to(wmap.pulls_for(goal + correction))
        if why is not None:
            return result(False, gap(tip), tip, why)
    tip = vision.measure_tip(3, on_frame) or tip
    e = gap(tip)
    near = not math.isnan(e) and e < tol_mm
    return result(near, e, tip, "reached" if near else
                  "out of reach" if not wmap.reachable(goal) else "no closer (friction or contact?)")


def follow(cables, vision, wmap, path, target, on_frame, spacing_mm=5.0, **kw):
    """Drive along a planned route given as tip points (x, z): each point every spacing_mm becomes
    pulls from the map (pulls_for), played back with `play`."""
    route = [np.asarray(p, float) for p in path]
    return play(cables, vision, wmap, [wmap.pulls_for(p) for p in densify(route, spacing_mm)], target, on_frame, **kw)
