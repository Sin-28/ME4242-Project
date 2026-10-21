# Optional rod bending display

This is a read-only overlay on the color and depth frames already acquired by
`nus_main.py`. It never changes motor commands, calibration, or route planning.
It is **off by default**.

## One-time setup

With the camera connected, run:

```bash
python nus_main.py bending-setup
```

This setup uses the D405 RGB feed through OpenCV and does not start the depth
SDK or motors. If the D405 is not camera index 0, add `--camera N`.

In the camera window, press `r` and draw an ROI that includes the green tip and
all nine blue discs **throughout their expected movement**. The fixed base mount
can stay outside the ROI. Press `a`, then click the point where the rod leaves
the mount. If the default colors do not detect the discs, press `b` and click
the center of one blue disc; press `g` and click the green tip. Press `q` to
finish. With all ten discs visible, press `s` to save their reference positions.
The settings are saved in ignored `bending_config.json` and reused when
the frame size matches. Run setup again if the camera or mount moves.

## Turn it on and off

Check the RGB tracking without motors or the depth SDK:

```bash
python nus_main.py bending-preview
```

To measure bends toward or away from the camera without motors, use the
aligned RealSense depth stream:

```bash
python nus_main.py bending-preview --depth
```

Before trusting 3D angles, check the aligned depth of each marker:

```bash
python nus_main.py bending-preview --depth-check
```

This shows a millimeter reading beside the green tip and every blue disc.
Move one marker toward and away from the camera: its reading should decrease
and increase respectively while stationary markers remain steady. `--` means
there were too few valid depth pixels on that colored disc or the readings
disagreed. Press `s` to save the annotated view and a CSV with depth, valid
pixel count, and depth spread in `runs/`. The 3D bending view uses these
color-gated readings rather than nearby background depth.

This uses 1280×720 aligned color/depth frames so the saved RGB setup can be
reused. On the test Mac the SDK crashes while starting either 640×480 or
1280×720 depth streams; the command checks this in a child process and reports
the failure without crashing the preview process. RGB alone cannot determine
front/back bending. Use a machine where the D405 depth stream starts normally
to validate 3D angles.

The RGB preview shows 2D angles only. The regular N-U-S commands use
`Vision.open_camera()` from `emioapi` for depth. On the test Mac, the D405
appeared on USB and RGB preview worked, but the RealSense SDK crashed during
device enumeration. The integrated 3D overlay needs a successful depth-camera
startup before it can be validated live.

Append `--bending` to any existing command, for example:

```bash
python nus_main.py detect --bending
python nus_main.py search --bending --record
```

During any camera session, press `v` to toggle the overlay. Without
`--bending`, it starts off. The existing `q` / Esc stop keys still work.

The overlay draws a numbered blue ring around each intermediate marker, a green
ring around the tip, a red base ring, and a yellow connecting line. It shows
each detected bend separately (`B1`, `B2`, etc.). Signed 2D angles use `+` for
clockwise motion on screen. With valid depth at every marker, it also estimates
unsigned 3D bend angles, including motion toward or away from the camera.
Missing depth leaves the 2D measurement and a depth status visible. Angles are
shown only when all ten markers are visible; held positions remain outlined
briefly during occlusion, with angle measurement paused. After tracking loss, the overlay retries the saved
reference pose and original marker colors automatically. The tracking area
extends to the saved base position even if the selected ROI ended before the
base-side blue marker. Optical flow predicts marker positions between frames so
fast bending can move beyond the local color search window.

The initial marker ordering assumes the base is to the right of the tip in the
image. Local tracking can follow a brief reversal in image x, but a full
reacquisition still needs the markers to separate across the image. The
dedicated `--depth` preview
aligns depth to color and deprojects each marker using RealSense intrinsics.
The regular N-U-S commands use the existing `emioapi` camera, which does not
explicitly align depth pixels to the color image. Their bending overlay shows
2D only until aligned depth is supplied, avoiding misleading 3D readings.
The repository tests use synthetic color and depth frames.
