#!/usr/bin/env python3
"""Bend the EMIO rod with arrow keys in a VS Code terminal."""

import argparse
import curses
import json
import math
import os
import sys
import time
from pathlib import Path

MOTOR_IDS = (0, 1, 2, 3)
FOLLOW_ERROR_DEGREES = 8.0
STALL_SECONDS = 1.5
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
EXTENDED_POSITION_MODE = 4
MIN_SIGNED_32 = -(1 << 31)
MAX_SIGNED_32 = (1 << 31) - 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", type=Path, metavar="FILE", help="save manual movements to JSON")
    mode.add_argument("--play", type=Path, metavar="FILE", help="replay saved movements")
    mode.add_argument("--session", action="store_true",
                      help="switch between manual control, recording, and playback without disconnecting")
    parser.add_argument("--up", type=int, choices=MOTOR_IDS, default=0)
    parser.add_argument("--down", type=int, choices=MOTOR_IDS, default=2)
    parser.add_argument("--left", type=int, choices=MOTOR_IDS, default=1)
    parser.add_argument("--right", type=int, choices=MOTOR_IDS, default=3)
    parser.add_argument(
        "--step", type=float, default=5.0,
        help="motor degrees per arrow press (default: 5)",
    )
    parser.add_argument(
        "--reverse", type=int, nargs="*", choices=MOTOR_IDS, default=[1, 3],
        help="motor IDs whose pull direction increases encoder pulses (default: 1 3)",
    )
    args = parser.parse_args()
    mapping = {name: getattr(args, name) for name in ("up", "down", "left", "right")}
    if len(set(mapping.values())) != 4:
        parser.error("up, down, left, and right must use four different motor IDs")
    if not math.isfinite(args.step) or not 0 < args.step <= 10:
        parser.error("--step must be greater than 0 and at most 10 degrees")
    return args, mapping


def load_recording(path, args, mapping):
    data = json.loads(path.read_text())
    recorded_mapping = data.get("mapping")
    recorded_reverse = data.get("reverse")
    if (data.get("version") != 1 or
            not isinstance(recorded_mapping, dict) or
            set(recorded_mapping) != set(mapping) or
            sorted(recorded_mapping.values()) != list(MOTOR_IDS) or
            not isinstance(recorded_reverse, list) or
            any(type(motor_id) is not int or motor_id not in MOTOR_IDS for motor_id in recorded_reverse)):
        raise ValueError("Invalid recording version or motor configuration")
    events = data.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("Recording has no movements")
    previous_time = -1.0
    for event in events:
        if not isinstance(event, dict) or set(event) != {"time", "x", "y"}:
            raise ValueError("Invalid recording event")
        t, x, y = event["time"], event["x"], event["y"]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (t, x, y)):
            raise ValueError("Recording contains a non-finite value")
        if t < 0 or t < previous_time:
            raise ValueError("Recording events are out of time order")
        previous_time = t
    return events, recorded_mapping, recorded_reverse


def save_recording(path, mapping, reverse, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps({
            "version": 1, "mapping": mapping, "reverse": sorted(reverse),
            "events": events,
        }, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def connect():
    from emioapi import EmioMotors
    from emioapi.emiomotors import motorgroup

    devices = motorgroup.listMotors()
    print(f"Detected EMIO devices: {devices}")
    if not devices:
        raise SystemExit("No EMIO controller found")
    motors = EmioMotors()
    if not motors.open(devices[0]):
        raise SystemExit(f"Could not connect to {devices[0]}")
    return motors


def signed32(value):
    """Interpret a Dynamixel 4-byte position register as a signed value."""
    value = int(value) & 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


def enter_extended_position_mode(motors):
    """Enable multi-turn position control without changing the current pose."""
    from dynamixel_sdk import COMM_SUCCESS

    mg = motors._mg
    port, packet = mg.portHandler, mg.packetHandler

    def check(action, motor_id, result, error):
        if result != COMM_SUCCESS or error != 0:
            raise RuntimeError(
                f"Motor {motor_id}: {action} failed "
                f"(communication={result}, status={error})"
            )

    for motor_id in MOTOR_IDS:
        check("torque off", motor_id, *packet.write1ByteTxRx(
            port, motor_id, ADDR_TORQUE_ENABLE, 0
        ))
        check("enable extended position mode", motor_id, *packet.write1ByteTxRx(
            port, motor_id, ADDR_OPERATING_MODE, EXTENDED_POSITION_MODE
        ))
        present, result, error = packet.read4ByteTxRx(
            port, motor_id, ADDR_PRESENT_POSITION
        )
        check("read position", motor_id, result, error)
        check("hold current position", motor_id, *packet.write4ByteTxRx(
            port, motor_id, ADDR_GOAL_POSITION, present
        ))
        check("torque on", motor_id, *packet.write1ByteTxRx(
            port, motor_id, ADDR_TORQUE_ENABLE, 1
        ))


def read_positions(motors):
    positions = motors._mg.getCurrentPosition()
    if positions is None or len(positions) != len(MOTOR_IDS):
        raise RuntimeError("Could not read all four motor positions")
    return [signed32(pulse) for pulse in positions]


def check_motor_health(motors):
    """Stop on a servo fault instead of treating its rejected goal as a readback glitch."""
    packet = motors._mg.packetHandler
    port = motors._mg.portHandler
    for motor_id in MOTOR_IDS:
        fault, result, error = packet.read1ByteTxRx(
            port, motor_id, ADDR_HARDWARE_ERROR
        )
        if result != 0:
            raise RuntimeError(f"Could not read motor {motor_id} hardware status: {result}")
        if fault:
            detail = "overload" if fault & 32 else f"error bits {fault}"
            raise RuntimeError(
                f"Motor {motor_id} hardware fault: {detail} (status {fault}). "
                "Stop moving the rod, relieve the load, and inspect the motor/cable "
                "before resetting power. No further motion was commanded."
            )
        if error:
            raise RuntimeError(f"Motor {motor_id} returned status error {error}")
        torque, result, error = packet.read1ByteTxRx(
            port, motor_id, ADDR_TORQUE_ENABLE
        )
        if result != 0 or error:
            raise RuntimeError(
                f"Could not read motor {motor_id} torque state: "
                f"communication={result}, status={error}"
            )
        if torque != 1:
            raise RuntimeError(
                f"Motor {motor_id} has torque disabled; no further motion was commanded"
            )


def goal_pulses(baseline, x, y, mapping, reverse, rad_to_pulse):
    """Convert horizontal/vertical bend amounts into four motor goals."""
    pull = {
        "up": max(y, 0.0),
        "down": max(-y, 0.0),
        "left": max(-x, 0.0),
        "right": max(x, 0.0),
    }
    goals = list(baseline)
    for direction, degrees in pull.items():
        motor_id = mapping[direction]
        pulse_change = round(math.radians(degrees) * rad_to_pulse)
        goals[motor_id] += pulse_change if motor_id in reverse else -pulse_change
    for motor_id, pulse in enumerate(goals):
        if not MIN_SIGNED_32 <= pulse <= MAX_SIGNED_32:
            raise ValueError(
                f"M{motor_id} goal {pulse}, start {baseline[motor_id]}; "
                "extended-position register limit exceeded"
            )
    return goals


def write_goals(motors, goals):
    """Send all four goals in one sync-write packet, then verify their registers."""
    writer = motors._mg.groupWriters["goal_position"]
    writer.clearParam()
    try:
        for motor_id, pulse in enumerate(goals):
            if not MIN_SIGNED_32 <= pulse <= MAX_SIGNED_32:
                raise ValueError(f"Motor {motor_id} goal {pulse} exceeds signed 32-bit position range")
            raw_position = pulse & 0xFFFFFFFF
            if not writer.addParam(motor_id, list(raw_position.to_bytes(4, "little"))):
                raise RuntimeError(f"Could not queue goal for motor {motor_id}")
        result = writer.txPacket()
        if result != 0:
            raise RuntimeError(f"EMIO sync write failed: communication={result}")
    finally:
        writer.clearParam()
    reported = motors._mg.getGoalPosition()
    if reported is None or [signed32(pulse) for pulse in reported] != goals:
        check_motor_health(motors)
        decoded = None if reported is None else [signed32(pulse) for pulse in reported]
        raise RuntimeError(f"Motor goal readback differs: sent {goals}, read {decoded}")


def draw(screen, mapping, baseline, positions, goals, x, y, step, message,
         rad_to_pulse, mode):
    screen.erase()
    travel = [
        (abs(goal - start) / rad_to_pulse * 180 / math.pi,
         abs(actual - start) / rad_to_pulse * 180 / math.pi)
        for start, actual, goal in zip(baseline, positions, goals)
    ]
    lines = [
        f"STATUS: {message}",
        "EMIO arrow control — rod center is the startup position",
        ("R: record/save  |  P: play/stop  |  Arrows: bend  |  Space: center  |  Q: exit"
         if mode == "session" else
         "Playback: Q or Esc stops and releases torque" if mode == "play" else
         "Arrows: bend  |  Space: return to center  |  Q or Esc: release and exit"),
        f"Mapping: ↑={mapping['up']}  ↓={mapping['down']}  ←={mapping['left']}  →={mapping['right']}",
        f"Offset: horizontal {x:+.1f}°, vertical {y:+.1f}°  |  step {step:g}°",
        "Travel from startup, command/measured: " + "  ".join(
            f"M{motor_id} {commanded:.0f}°/{measured:.0f}°"
            for motor_id, (commanded, measured) in enumerate(travel)
        ),
        f"Startup pulses: {baseline}",
        f"Current pulses: {positions}",
        f"Goal pulses:    {goals}",
    ]
    height, width = screen.getmaxyx()
    for row, line in enumerate(lines[:height]):
        if width > 1:
            screen.addnstr(row, 0, line, width - 1)
    screen.refresh()


def center_rod(motors, baseline, x, y, mapping, reverse, step):
    steps = max(1, math.ceil(max(abs(x), abs(y)) / step))
    for index in range(1, steps + 1):
        fraction = 1 - index / steps
        goals = goal_pulses(
            baseline, x * fraction, y * fraction, mapping, reverse, motors._rad_to_pulse
        )
        write_goals(motors, goals)
        time.sleep(0.02)
    return list(baseline)


def numbered_motion_path(key):
    if ord("0") <= key <= ord("9"):
        return Path(f"motion_{chr(key)}.json")
    return None


def keyboard_loop(screen, motors, baseline, args, mapping, events=None, recording=None,
                  session_state=None):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    screen.timeout(30)
    reverse = set(args.reverse)
    goals = list(baseline)
    positions = list(baseline)
    last_motion_pulse = list(baseline)
    last_motion_time = [time.monotonic()] * len(MOTOR_IDS)
    last_goal_time = list(last_motion_time)
    last_health_check = time.monotonic()
    follow_error = round(math.radians(FOLLOW_ERROR_DEGREES) * motors._rad_to_pulse)
    x = y = 0.0
    message = "Press an arrow key to move one step."
    if recording is None:
        recording = []
    if session_state is None:
        session_state = {}
    started = time.monotonic()
    play_index = 0
    play_start = started
    session_events = None
    session_play_origin = (0.0, 0.0)
    live_mapping = mapping
    live_reverse = set(reverse)
    session_record_path = None
    session_record_origin = (0.0, 0.0)
    session_record_started = started
    awaiting_slot = None
    if events is not None:
        message = "Playing recording; Q or Esc stops."

    while True:
        positions = read_positions(motors)
        now = time.monotonic()
        if now - last_health_check >= 0.5:
            check_motor_health(motors)
            last_health_check = now
        for motor_id in MOTOR_IDS:
            if abs(positions[motor_id] - last_motion_pulse[motor_id]) >= 2:
                last_motion_pulse[motor_id] = positions[motor_id]
                last_motion_time[motor_id] = now
            error = abs(goals[motor_id] - positions[motor_id])
            if error > follow_error and now - max(
                last_motion_time[motor_id], last_goal_time[motor_id]
            ) > STALL_SECONDS:
                raise RuntimeError(
                    f"Motor {motor_id} stopped {error} pulses short of its goal; "
                    "keyboard control stopped and torque was released"
                )

        draw(
            screen, mapping, baseline, positions, goals, x, y,
            args.step, message, motors._rad_to_pulse,
            "session" if args.session else "play" if events is not None else
            "record" if args.record else "manual",
        )
        key = screen.getch()
        if key in (ord("q"), ord("Q"), 27):
            if session_record_path is not None:
                save_recording(session_record_path, mapping, args.reverse, recording)
                session_state.pop("record_path", None)
            return recording
        if args.session and awaiting_slot is not None:
            path = numbered_motion_path(key)
            operation = awaiting_slot
            if path is None:
                if key != -1:
                    awaiting_slot = None
                    message = "Selection cancelled; press R or P, then a digit."
                continue
            awaiting_slot = None
            if operation == "record":
                session_record_path = path
                session_state["record_path"] = path
                session_record_origin = (x, y)
                session_record_started = time.monotonic()
                recording.clear()
                message = f"Recording to {path}; press R to save."
            else:
                try:
                    loaded, saved_mapping, saved_reverse = load_recording(path, args, mapping)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    message = f"Cannot play {path}: {exc}"
                else:
                    baseline = list(positions)
                    goals = list(positions)
                    write_goals(motors, goals)
                    x = y = 0.0
                    mapping = saved_mapping
                    reverse = set(saved_reverse)
                    session_events = loaded
                    session_play_origin = (0.0, 0.0)
                    play_index = 0
                    play_start = time.monotonic()
                    message = f"Playing {path}; press P to stop."
            continue
        if args.session and key in (ord("r"), ord("R")) and session_events is None:
            if session_record_path is not None:
                save_recording(session_record_path, mapping, args.reverse, recording)
                session_state.pop("record_path", None)
                message = f"Saved {len(recording)} movements to {session_record_path}."
                session_record_path = None
            else:
                awaiting_slot = "record"
                message = "Press digit 0–9 to choose motion_N.json for recording."
            continue
        if args.session and key in (ord("p"), ord("P")) and session_record_path is None:
            if session_events is not None:
                session_events = None
                baseline = list(positions)
                goals = list(positions)
                write_goals(motors, goals)
                x = y = 0.0
                mapping = live_mapping
                reverse = set(live_reverse)
                message = "Playback stopped; manual control resumed."
            else:
                awaiting_slot = "play"
                message = "Press digit 0–9 to play motion_N.json."
            continue
        active_events = session_events if session_events is not None else events
        if active_events is not None:
            if play_index >= len(active_events):
                if all(abs(goal - actual) <= follow_error for goal, actual in zip(goals, positions)):
                    if session_events is not None:
                        session_events = None
                        baseline = list(positions)
                        goals = list(positions)
                        write_goals(motors, goals)
                        x = y = 0.0
                        mapping = live_mapping
                        reverse = set(live_reverse)
                        message = "Playback complete; manual control resumed."
                        continue
                    return recording
                continue
            event = active_events[play_index]
            if time.monotonic() - play_start < event["time"]:
                continue
            next_x = session_play_origin[0] + event["x"] if session_events is not None else event["x"]
            next_y = session_play_origin[1] + event["y"] if session_events is not None else event["y"]
            key = None
        else:
            next_x, next_y = x, y
        if key == ord(" "):
            previous_goals = goals
            goals = center_rod(motors, baseline, x, y, mapping, reverse, args.step)
            for motor_id in MOTOR_IDS:
                if goals[motor_id] != previous_goals[motor_id]:
                    last_goal_time[motor_id] = time.monotonic()
            x = y = 0.0
            message = "Returning to startup position."
            if args.record or session_record_path is not None:
                if session_record_path is not None:
                    recording.append({"time": time.monotonic() - session_record_started,
                                      "x": x - session_record_origin[0],
                                      "y": y - session_record_origin[1]})
                else:
                    recording.append({"time": time.monotonic() - started, "x": x, "y": y})
            continue

        if key == curses.KEY_UP:
            next_y = y + args.step
        elif key == curses.KEY_DOWN:
            next_y = y - args.step
        elif key == curses.KEY_LEFT:
            next_x = x - args.step
        elif key == curses.KEY_RIGHT:
            next_x = x + args.step
        elif active_events is None:
            continue
        if (next_x, next_y) == (x, y):
            if active_events is not None:
                play_index += 1
            message = "Position unchanged."
            continue
        lagging = [
            motor_id for motor_id in MOTOR_IDS
            if abs(goals[motor_id] - positions[motor_id]) > follow_error
        ]
        if lagging:
            message = (
                f"Waiting for motor(s) {', '.join(map(str, lagging))} to catch up. "
                "Space returns to center."
            )
            continue
        try:
            next_goals = goal_pulses(
                baseline, next_x, next_y, mapping, reverse, motors._rad_to_pulse
            )
        except ValueError as exc:
            message = str(exc)
            continue
        write_goals(motors, next_goals)
        for motor_id in MOTOR_IDS:
            if next_goals[motor_id] != goals[motor_id]:
                last_goal_time[motor_id] = time.monotonic()
        goals = next_goals
        x, y = next_x, next_y
        if active_events is not None:
            play_index += 1
            message = f"Playing {play_index}/{len(active_events)}"
        elif args.record or session_record_path is not None:
            if session_record_path is not None:
                recording.append({"time": time.monotonic() - session_record_started,
                                  "x": x - session_record_origin[0],
                                  "y": y - session_record_origin[1]})
            else:
                recording.append({"time": time.monotonic() - started, "x": x, "y": y})
            message = f"Recorded {len(recording)} movement(s)."
        else:
            message = ""


def main():
    args, mapping = parse_args()
    events = None
    if args.play:
        events, mapping, args.reverse = load_recording(args.play, args, mapping)
    recording = []
    session_state = {}
    motors = connect()
    try:
        check_motor_health(motors)
        enter_extended_position_mode(motors)
        check_motor_health(motors)
        baseline = read_positions(motors)
        print(f"Rod center recorded from startup pulses: {baseline}")
        write_goals(motors, baseline)
        curses.wrapper(keyboard_loop, motors, baseline, args, mapping, events, recording,
                       session_state)
    finally:
        try:
            motors.close()
        except Exception as exc:
            print(f"Could not close the EMIO connection cleanly: {exc}", file=sys.stderr)
        if args.record:
            save_recording(args.record, mapping, args.reverse, recording)
            print(f"Saved {len(recording)} movements to {args.record}")
        if session_state.get("record_path"):
            path = session_state["record_path"]
            save_recording(path, mapping, args.reverse, recording)
            print(f"Saved {len(recording)} movements to {path}")


if __name__ == "__main__":
    main()
