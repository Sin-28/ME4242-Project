# Optional 2D rod bending display

The bending overlay uses the RGB camera to track the green tip and nine blue
markers. It draws the numbered markers, a yellow connection line to the fixed
base, and labels bends greater than 5° beside the rod. It is read-only and off
by default during the existing N-U-S commands. This branch does not use depth
for bending tracking or 3D bend angles. Similar-colored objects can still
confuse RGB tracking.

## Calibrate the markers

```bash
python nus_main.py bending-setup
```

Add `--camera N` if the RGB camera is not index 0. In the setup window:

1. Press `r` and draw an ROI covering all expected marker movement. Include
   the green tip and all blue markers; keep similar-colored background objects
   outside it where possible.
2. Press `a` and click the fixed base.
3. Press `g` and click the green tip center.
4. Press `b` once and click blue markers B1 through B9 from tip toward base.
   Each click displays a numbered ring. Press `u` to undo a marker or `g` to
   restart the sequence.
5. Press `s` to save and `q` to close.

The setup saves `bending_config.json` locally. The RGB camera requests its saved
frame size before capturing, and the overlay reloads settings when the actual
frame size matches. A new ROI clears the saved marker sequence.

## Preview

```bash
python nus_main.py bending-preview
```

Press `m` to reopen setup, `v` to toggle the overlay, and `q` or Esc to quit.
The preview does not start the motors. If tracking loses the chain, the saved
reference rings remain visible so that the expected and observed locations can
be compared. Marker identities follow the manually clicked order. The tip and
connection line update with tracked positions, and the two largest signed 2D
bends over 5° are labeled on the image. Angle labels pause while any marker is
held after a missed detection; recalibrate if the reference rings no longer
match the rod.
After a large bend, the overlay automatically looks for the last fully tracked
pose and the saved pose, including a shifted return position. A full ordered
chain is required before angle labels resume. This recovery updates the live
tracker only; it does not overwrite manually saved calibration clicks.
If two search windows see the same disc, the tracker holds the ambiguous
identity and reacquires the chain instead of reporting an overlap angle error.

The overlay can also be enabled for existing commands, for example:

```bash
python nus_main.py detect --bending
python nus_main.py search --bending --record
```

The existing N-U-S vision and motor functions have their own camera requirements.
The bending overlay in this branch uses their RGB frame only. Marker ordering
assumes the fixed base is to the right of the tip and the rod does not fold
backward in image x.
