# Optional rod bending display

This is a read-only overlay on the color and depth frames already acquired by
`nus_main.py`. It never changes motor commands, calibration, or route planning.
It is **off by default**.

## One-time setup

With the camera connected, run:

```bash
python nus_main.py bending-setup
```

In the camera window, press `r` and draw an ROI that includes the green tip and
all nine blue discs **throughout their expected movement**. The fixed base mount
can stay outside the ROI. Press `a`, then click the point where the rod leaves
the mount. If the default colors do not detect the discs, press `b` and click
the center of one blue disc; press `g` and click the green tip. Press `q` to
finish. The settings are saved in ignored `bending_config.json` and reused when
the frame size matches. Run setup again if the camera or mount moves.

## Turn it on and off

Append `--bending` to any existing command, for example:

```bash
python nus_main.py detect --bending
python nus_main.py search --bending --record
```

During any camera session, press `v` to toggle the overlay. Without
`--bending`, it starts off. The existing `q` / Esc stop keys still work.

The overlay shows signed 2D angles (`+` is clockwise on screen). When depth is
available at the base and all visible discs, it also shows unsigned 3D angles.
For two opposite bends, it displays base-side and tip-side angles separately.
If depth is missing, the 2D measurement remains visible with a depth status.
If fewer than seven colored discs are found, no angle is reported.

The marker ordering assumes the base is to the right of the tip in the image
and the rod does not fold backward in image x. 3D values use the existing
`emioapi` camera calibration and its millimeter depth frame. Live accuracy
needs checking with the physical camera and marker arrangement; the repository
tests use synthetic color and depth frames because hardware was unavailable.
