"""Aligned RealSense color/depth frames via the installed Homebrew SDK."""
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import time

import cv2
import numpy as np


class AlignedDepthCamera:
    """Use Homebrew librealsense, avoiding the older macOS Python wheel."""

    def __init__(self):
        directory = Path(__file__).resolve().parent
        source = directory / "nus_depth_bridge.cpp"
        binary = directory / ".build" / "nus_depth_bridge"
        if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
            binary.parent.mkdir(exist_ok=True)
            prefix = Path("/opt/homebrew/opt/librealsense")
            command = ["clang++", "-std=c++17", "-O2", str(source), "-o", str(binary),
                       f"-I{prefix / 'include'}", f"-L{prefix / 'lib'}",
                       f"-Wl,-rpath,{prefix / 'lib'}", "-lrealsense2"]
            built = subprocess.run(command, capture_output=True, text=True)
            if built.returncode:
                raise RuntimeError("Could not build RealSense bridge: " + built.stderr.strip())
        # macOS 12+ requires elevated libusb access to the D405 depth sensor.
        # Keep the GUI as the normal user and elevate only this camera bridge.
        command = ["sudo", "-n", str(binary)] if sys.platform == "darwin" else [str(binary)]
        print(f"D405 bridge: {binary} (administrator USB access)", flush=True)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        bufsize=0)
        self.frame = self.depth_frame = None
        self.intrinsics = None
        self._first_frame = True
        self._bridge_status = "starting bridge"
        self._stderr_buffer = bytearray()
        self._stderr_lines = []

    def _read_status(self):
        descriptor = self.process.stderr.fileno()
        data = os.read(descriptor, 4096)
        if not data:
            return
        self._stderr_buffer.extend(data)
        while b"\n" in self._stderr_buffer:
            line, _, rest = self._stderr_buffer.partition(b"\n")
            self._stderr_buffer = bytearray(rest)
            message = line.decode(errors="replace").strip()
            if message:
                self._bridge_status = message
                self._stderr_lines.append(message)
                self._stderr_lines = self._stderr_lines[-30:]
                if self._first_frame and message.startswith("stage:"):
                    print("D405 " + message, flush=True)

    def _failure_detail(self, remaining=""):
        lines = self._stderr_lines.copy()
        if remaining:
            lines.extend(remaining.splitlines())
        details = [line for line in lines if not line.startswith("stage:")]
        if details:
            return self._explain_error("\n".join(details[-20:]))
        return self._bridge_status

    @staticmethod
    def _explain_error(detail):
        if "password is required" in detail:
            return ("D405 depth access requires administrator permission. "
                    "Run 'sudo -v' in this Terminal, then start bending-preview again.")
        return detail

    def _read_exact(self, count, deadline):
        result = bytearray()
        descriptor = self.process.stdout.fileno()
        status_descriptor = self.process.stderr.fileno()
        while len(result) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"No aligned RealSense frame ({self._failure_detail()}); "
                                   "check D405 USB connection and close other camera apps")
            ready, _, _ = select.select([descriptor, status_descriptor], [], [], remaining)
            if status_descriptor in ready:
                self._read_status()
            if descriptor not in ready:
                continue
            chunk = os.read(descriptor, count - len(result))
            if not chunk:
                self.process.wait(timeout=2)
                detail = self.process.stderr.read().decode(errors="replace").strip()
                reason = self._failure_detail(detail)
                raise RuntimeError(f"{reason} (bridge exit {self.process.returncode})")
            result.extend(chunk)
        return result

    def update(self):
        try:
            self.process.stdin.write(b"R")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            detail = self.process.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(self._failure_detail(detail) or "RealSense bridge stopped") from exc
        deadline = time.monotonic() + (20 if self._first_frame else 8)
        width, height, fx, fy, cx, cy, scale_mm = struct.unpack(
            "<II5f", self._read_exact(28, deadline))
        if not 1 <= width <= 1920 or not 1 <= height <= 1080:
            raise RuntimeError("Invalid RealSense frame dimensions")
        color = self._read_exact(width * height * 2, deadline)
        depth = self._read_exact(width * height * 2, deadline)
        yuyv = np.frombuffer(color, np.uint8).reshape(height, width, 2)
        self.frame = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        self.depth_frame = np.frombuffer(depth, np.uint16).reshape(height, width).astype(np.float32) * scale_mm
        self.intrinsics = (fx, fy, cx, cy)
        if self._first_frame:
            print(f"D405 first aligned frame: {width}x{height}, depth scale {scale_mm:.4f} mm", flush=True)
        self._first_frame = False

    def pixel_to_robot(self, x, y, z):
        fx, fy, cx, cy = self.intrinsics
        return ((x-cx)*z/fx, (y-cy)*z/fy, z)

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b"Q")
                self.process.stdin.close()
                self.process.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=2)
        self.process.stdout.close()
        self.process.stderr.close()
