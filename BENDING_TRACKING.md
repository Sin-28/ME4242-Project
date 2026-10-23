# Optional rod bending display

This is a read-only overlay on the color and depth frames already acquired by
`nus_main.py`. It never changes motor commands, calibration, or route planning.
It is **off by default**.

## One-time setup

With the camera connected, run:

```bash
python nus_main.py bending-setup
```

This setup uses aligned D405 RGB and depth through the Homebrew `librealsense`
SDK (2.59.1 on the test Mac) and does not start the motors. The small C++ camera
bridge is compiled automatically with `clang++` on first use. Use `--rgb-only`
for the older color-only setup; in that mode, add
`--camera N` if the D405 is not OpenCV camera index 0.

In the camera window, press `r` and draw an ROI that includes the green tip and
all nine blue discs throughout their expected movement. Press `a` and click the
fixed base. Press `g` and click the green tip **center**. Then press `b` once
and click each blue disc center in order from the tip toward the base (`B1`
through `B9`, left to right). Each click immediately shows a numbered ring.
Press `u` to undo a mistaken point, or `g` to start the marker sequence again.
Press `s` to save and `q` to finish. The clicks also sample the green and first
blue color when possible; washed-out color does not discard a manually placed
point. A new ROI clears the prior marker sequence. Depth setup saves
`bending_depth_config.json`; RGB-only setup saves `bending_config.json`.
Settings are reused when the frame size matches.

## Depth-validated marker tracking

On macOS 12+, librealsense USB depth access may require administrator privileges.
On the test Mac, RGB opened without them but the SDK reported `cannot access depth
sensor`. In the same Terminal used to run this application, first run `sudo -v`
and enter the Mac administrator password. Then run the normal `python nus_main.py
bending-preview` command. The application runs as the normal user and only its
small camera bridge uses the cached administrator permission. The SDK can be
checked independently with `sudo /opt/homebrew/bin/rs-enumerate-devices`.
See the [RealSense macOS installation instructions](https://github.com/realsenseai/librealsense/blob/master/doc/installation_osx.md).

Close any open RGB preview, then calibrate the D405 with aligned color and depth
using the setup command above. Click the ROI, base, green tip and nine blue
discs. The depth setup
saves a separate `bending_depth_config.json` at 640 × 480. It must report
"Saved 10 markers with aligned depth"; if a marker has no depth, adjust the
camera or object and press `s` again. Then run:

```bash
python nus_main.py bending-preview
```

This mode uses color, previous position and each marker's measured depth.
The D405 bridge requests simultaneous YUYV color and Z16 depth at 640 × 480,
30 fps, then tries supported lower frame rates and resolutions if the SDK
cannot resolve that pair. It can open the two sensor profiles directly if the
pipeline resolver rejects them. YUYV is converted to BGR for the overlay.
Pixels more than 40 mm from that marker's recent depth, or pixels with no
valid depth, cannot move the marker. The D405 depth image is aligned to its
color image before tracking. If aligned frames are unavailable, the command
stops with an error rather than silently using color alone. Live depth capture
on the test Mac is currently blocked below the application: the D405 is
detected through a GenesysLogic USB hub, but the Homebrew librealsense 2.59.1
bridge cannot start even a Z16-only stream. The official `rs-depth` example
reports that the detected D405 does not support depth streaming, and
`rs-fw-update -b` fails with `cannot access depth sensor` even with
administrator access. The older Python SDK also detects the device but crashes
while starting a depth-only stream. Do not rely on depth mode until an
independent RealSense example captures depth frames. Test a different USB 3
connection or host when available. Firmware updating is not recommended while
the depth interface is inaccessible and a backup cannot be made.

## RGB-only fallback and N-U-S overlay

Check the RGB-only tracking without motors or the depth SDK:

```bash
python nus_main.py bending-preview --rgb-only
```

The RGB preview shows 2D angles only and cannot reject a same-colored object
using depth. Press `m` to open the manual setup
screen while previewing; returning to preview reloads the saved points. For
RGB-only manual setup, use `python nus_main.py bending-setup --rgb-only`. If
color tracking is lost, the saved reference rings remain visible so you can
compare the expected marker locations with the camera image. The regular
N-U-S commands use
`Vision.open_camera()` from `emioapi` for depth. On the test Mac, the D405
appeared on USB and RGB preview worked, but the older Python RealSense SDK
crashed during stream startup. The existing N-U-S overlay receives depth for 3D angle calculation, but its
`emioapi` color and depth frames have no guaranteed pixel alignment. It still
uses RGB for marker identification; use the default `bending-preview` for depth-
validated tracking until that camera path is aligned.

Append `--bending` to any existing command, for example:

```bash
python nus_main.py detect --bending
python nus_main.py search --bending --record
```

During any camera session, press `v` to toggle the overlay. Without
`--bending`, it starts off. The existing `q` / Esc stop keys still work.

The overlay draws a numbered blue ring around each intermediate marker, a green
ring around the tip, a red base ring, and a yellow connecting line. It shows
each detected bend separately (`B1`, `B2`, etc.). The green tip and yellow
base-to-tip connection update with the tracked markers. Any 2D bend over 5°
is also labeled beside its measured section. Signed 2D angles use `+` for
clockwise motion on screen. With valid depth at every marker, it also estimates
unsigned 3D bend angles, including motion toward or away from the camera.
Missing depth leaves the 2D measurement and a depth status visible. Marker
tracking uses a small window around each marker's last known position. If
tracking is lost for several frames, it retries the saved reference pose and
original marker colors. The saved reference rings are displayed when the full
chain cannot be found. Keep the ROI around the moving markers and away from
similarly colored background objects; press `m` in RGB preview to adjust it.

The marker ordering assumes the base is to the right of the tip in the image
and the rod does not fold backward in image x. 3D values use the existing
`emioapi` camera calibration and its millimeter depth frame. The current SDK
returns depth and color frames without explicit pixel alignment, so the 3D
values are provisional until alignment and physical accuracy are verified.
The repository tests use synthetic color and depth frames.
