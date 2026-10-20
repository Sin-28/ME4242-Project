"""
EMIO hardware smoke test (ME4242 Project 2 - cable-driven manipulator).

Checks, in order:
  1. The EMIO motor board (U2D2 / FTDI serial port) and RealSense camera are detected.
  2. The 4 servo motors connect and report their angles.
  3. One motor makes a small, slow move and returns to where it started.
  4. The camera streams: saves a colour frame + depth image, prints green-marker positions.

Usage (inside the ME4242 conda env):
    python emio_hardware_test.py                     # full test, asks before moving
    python emio_hardware_test.py --no-move           # read-only: never moves a motor
    python emio_hardware_test.py --motor 2 --delta 0.1
    python emio_hardware_test.py --skip-camera
    python emio_hardware_test.py --skip-motors

Notes:
  - Motor angles are in radians. The servos accept goals within one turn, [-pi, pi]
    (pulse 0-4095). A jog that would leave that range is refused, never clamped:
    clamping can turn a small jog into a large move.
  - Pulley radius is 20 mm, so cable displacement (mm) = angle (rad) x 20.
    The default jog of 0.10 rad is about 2 mm of cable.
  - Closing the motor connection disables torque, so the manipulator goes slack at the end.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import serial.tools.list_ports

import emioapi
from emioapi import EmioCamera, EmioMotors
from emioapi._motorgroup import listMotors

PULLEY_RADIUS_MM = 20.0          # from emioapi: _length_to_rad = 1/20
ANGLE_MIN, ANGLE_MAX = -math.pi, math.pi   # one servo turn (pulse 4095..0)
PROFILE_VELOCITY = 60            # Dynamixel units of 0.229 rev/min -> ~14 rev/min (slow)
MOVE_TIMEOUT_S = 4.0
CAMERA_WARMUP_FRAMES = 30        # let auto-exposure settle
OUTPUT_DIR = Path(__file__).resolve().parent / "test_output"

results = {}


def header(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def fmt_angles(angles):
    return "  ".join(
        f"M{i}: {a:+.3f} rad ({math.degrees(a):+7.2f} deg, cable {a * PULLEY_RADIUS_MM:+6.1f} mm)"
        for i, a in enumerate(angles)
    )


def wait_until_stopped(motors):
    """Poll the 'moving' flag until all motors stop or the timeout expires."""
    start = time.time()
    time.sleep(0.1)
    while time.time() - start < MOVE_TIMEOUT_S:
        moving = motors.moving
        if moving is not None and not any(moving):
            return True
        time.sleep(0.05)
    return False


def check_devices():
    header("1. Device detection")
    ports = list(serial.tools.list_ports.comports())
    print("Serial ports found:")
    for p in ports:
        print(f"  {p.device}: {p.description} (manufacturer: {p.manufacturer})")
    if not ports:
        print("  (none)")

    emio_ports = listMotors()
    print(f"EMIO motor ports (FTDI): {emio_ports or 'NONE'}")

    try:
        cameras = EmioCamera.listCameras()
    except Exception as e:
        cameras = []
        print(f"Camera listing failed: {e}")
    print(f"RealSense cameras: {cameras or 'NONE'}")

    results["motor board detected"] = bool(emio_ports)
    results["camera detected"] = bool(cameras)
    return bool(emio_ports), bool(cameras)


def test_motors(motor_id, delta, allow_move, assume_yes):
    header("2. Motors")
    motors = EmioMotors()
    if motors.findAndOpen() < 0:
        print("FAILED to connect to the motors. Check USB cable, motor power supply, "
              "and that EMIO Labs is closed (only one program can use the port).")
        results["motors connect"] = False
        return

    results["motors connect"] = True
    try:
        print(f"Connected on {motors.device_name}")
        try:
            start = motors.angles
        except Exception as e:
            start = None
            print(f"Read error: {e}")
        if start is None:
            print("FAILED to read motor positions. The USB board answers but the motors don't: "
                  "check the EMIO power supply is plugged in and switched on.")
            results["motors read"] = False
            return
        results["motors read"] = True
        print("Start position:\n  " + fmt_angles(start).replace("  M", "\n  M"))

        # Hold the current position before anything else, and move slowly.
        motors.max_velocity = [PROFILE_VELOCITY] * 4
        motors.angles = start

        if not allow_move:
            print("\n--no-move given: skipping the jog test.")
            return

        target = list(start)
        target[motor_id] = start[motor_id] + delta
        if not ANGLE_MIN <= target[motor_id] <= ANGLE_MAX:
            print(f"Jog refused: {target[motor_id]:+.3f} rad is outside the servo range "
                  f"[{ANGLE_MIN:+.3f}, {ANGLE_MAX:+.3f}]. Try the opposite --delta.")
            results["motor jog"] = False
            return

        print(f"\nPlanned jog: motor {motor_id} from {start[motor_id]:+.3f} to {target[motor_id]:+.3f} rad "
              f"(~{(target[motor_id] - start[motor_id]) * PULLEY_RADIUS_MM:+.1f} mm of cable), then back.")
        if not assume_yes:
            answer = input("Keep hands clear of the manipulator. Move now? [y/N] ").strip().lower()
            if answer != "y":
                print("Jog skipped.")
                return

        motors.angles = target
        wait_until_stopped(motors)
        reached = motors.angles
        error = abs(reached[motor_id] - target[motor_id])
        print(f"Reached: {reached[motor_id]:+.3f} rad (error {error:.3f} rad)")

        motors.angles = start
        wait_until_stopped(motors)
        back = motors.angles
        print(f"Returned: {back[motor_id]:+.3f} rad (start was {start[motor_id]:+.3f})")

        # 0.05 rad ~ 1 mm of cable; generous because cable tension loads the servo.
        results["motor jog"] = error < 0.05 and abs(back[motor_id] - start[motor_id]) < 0.05
        if not results["motor jog"]:
            print("Jog error is large: the cable may be too tight, or the motor is stalled.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        motors.close()
        print("Motors closed (torque off).")


def test_camera():
    header("3. Camera")
    camera = EmioCamera(show=False, track_markers=True, compute_point_cloud=False)
    if not camera.open():
        print("FAILED to open the camera. Check the USB cable (use a USB 3 port) and that "
              "EMIO Labs is closed.")
        results["camera stream"] = False
        return

    try:
        print(f"Camera {camera.camera_serial} opened, warming up...")
        for _ in range(CAMERA_WARMUP_FRAMES):
            camera.update()

        frame, depth = camera.frame, camera.depth_frame
        if frame is None or depth is None:
            print("FAILED to grab frames.")
            results["camera stream"] = False
            return
        results["camera stream"] = True

        OUTPUT_DIR.mkdir(exist_ok=True)
        color_path = OUTPUT_DIR / "camera_color.png"
        depth_path = OUTPUT_DIR / "camera_depth.png"
        cv2.imwrite(str(color_path), frame)
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(np.clip(depth, 0, camera.depth_max), alpha=255.0 / camera.depth_max),
            cv2.COLORMAP_JET,
        )
        cv2.imwrite(str(depth_path), depth_vis)
        print(f"Saved {color_path.name} and {depth_path.name} to {OUTPUT_DIR}")

        h, w = depth.shape
        print(f"Frame size {w}x{h}; depth at image centre: {int(depth[h // 2, w // 2])} mm "
              f"(0 = no reading)")

        markers = camera.trackers_pos
        print(f"Green markers tracked: {len(markers)}")
        for i, p in enumerate(markers):
            print(f"  marker {i}: x={p[0]:.1f}, y={p[1]:.1f}, z={p[2]:.1f}")
        if not markers:
            print("  (none - expected if no green marker / green tip is in view)")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.close()
        print("Camera closed.")


def main():
    parser = argparse.ArgumentParser(description="EMIO hardware smoke test")
    parser.add_argument("--motor", type=int, default=0, choices=range(4), help="motor to jog (0-3)")
    parser.add_argument("--delta", type=float, default=0.10, help="jog size in rad (default 0.10 ~ 2 mm cable)")
    parser.add_argument("--no-move", action="store_true", help="never move a motor")
    parser.add_argument("--yes", action="store_true", help="do not ask before moving")
    parser.add_argument("--skip-motors", action="store_true")
    parser.add_argument("--skip-camera", action="store_true")
    args = parser.parse_args()

    if abs(args.delta) > 0.5:
        sys.exit("--delta above 0.5 rad (~10 mm of cable) is too large for a first test.")

    print(f"Python {sys.version.split()[0]}, emioapi at {Path(emioapi.__file__).parent}")
    has_motors, has_camera = check_devices()

    if not args.skip_motors:
        if has_motors:
            test_motors(args.motor, args.delta, not args.no_move, args.yes)
        else:
            print("\nSkipping motor test: no EMIO motor board found.")
    if not args.skip_camera:
        if has_camera:
            test_camera()
        else:
            print("\nSkipping camera test: no RealSense camera found.")

    header("Summary")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(0 if results and all(results.values()) else 1)


if __name__ == "__main__":
    main()
