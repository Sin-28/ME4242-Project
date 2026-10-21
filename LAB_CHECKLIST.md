# Lab checklist: N -> U -> S search (no obstacles)

Run everything in the `ME4242` conda env from `C:\DEV Folder\ME4242 Project`, with **EMIO Labs closed**.
In any camera window, **q / Esc stops** and switches the motors off.

## 0. Setup (5 min)
- [ ] EMIO USB **firmly** plugged in + power on (the board has dropped out before).
- [ ] Coloured letters on the white platform (or white letters on black cloth, see step 3),
      **no green objects** in view, hands out of view.
- [ ] Arm hangs straight and freely.
- [ ] `conda activate ME4242`

## 0b. Check the set (every new set / new day, 1 min, motors don't move)
```
python nus_main.py check --apply
```
Saves this set's surface height (`stage_y_mm`). Read the "Tip" line:
- **OK / a bit off-centre**: carry on.
- **FAR off-centre while the arm hangs straight**: this set's camera is mounted differently.
  Do the camera calibration: arm out of view, ArUco marker (or the printed
  `aruco_672_calibration_marker.pdf`) centred on the platform, arrow toward the camera, then
  `python -m emioapi calibrate`. Then run `check --apply` again.

## 1. Cables taut (only if calibration complains about slack)
```
python nus_main.py jog
```
`0`-`3` select motor, `w`/`s` = +1/-1 mm, `W`/`S` = +5/-5 mm. Wind each cable until the
"tip moved" readout starts to rise (~5 px). **Write down which key tightened each motor**
(w = +1, s = -1):  M0 ___  M1 ___  M2 ___  M3 ___

## 2. Calibrate (every session and every set, ~2 min)
```
python nus_main.py calibrate --pull-signs=M0,M1,M2,M3      e.g. --pull-signs=-1,-1,1,1
```
(or plain `python nus_main.py calibrate` if you don't know the directions yet; on a
**different set** the directions can differ, so don't reuse another set's signs).
Good result: 4 lines "Motor j: pulls with direction ...", and the four "direction ... deg"
values roughly 90 deg apart, each moving the tip > 0.2 mm per mm of cable.
**Bad result:** "WARNING: motors a and b measured exactly the same", or all directions within one
half-circle (e.g. all near +90 or -90 deg). Then a cable is slack or the tip was stuck: check the
cables are taut, nudge the tip free, calibrate again. "the tip didn't come back by .. mm"
= friction with the platform; OK, the search re-measures the cables at every letter anyway.

## 3. Check the letters (camera only)
```
python nus_main.py detect
```
**Coloured letters (default, `letter_mode: "color"`): N red, U black, S blue, on the white platform.**
Teach each letter once per lighting setup: **click ON its paint (a stroke, not a gap) and press
n / u / s**. That learns its colour (saved in `nus_config.json`) and stores its shape
(`letter_templates_color.npz`). Do 2 examples per letter at different spots/angles.
All three should then be green, inside the grey circle, 1-2 cm apart, clear of the arm.
A red label like `S? 1.00` = right colour but rejected; the console/overlay reason says why
(round blob = arm disk, thin = cable, "mm above the letters" = on the arm).
(White letters on the black cloth: set `letter_mode` to `"shape"` and `background_hsv` to
`[[[0, 0, 0], [180, 255, 100]]]`.)

**Obstacles** show in the same window as **orange outlines** with their height ("25mm"), and the
line `obstacles (orange): 2 - 25 mm tall, 40 mm tall`. They are found by **height** with the depth
camera, whatever their colour: anything standing more than `obstacle_height_mm` (default 8) above
the platform (a plane fitted to the white platform) that isn't a letter. The arm is left out
because it reaches up to the robot (more than `obstacle_max_height_mm`, default 150, above the
platform), as are blobs touching the green tip. Clicking a pixel prints its height above the
platform, to check. Each obstacle should have one outline around its footprint; the arm, the
letters, shadows and marks on the platform should have none.

## 4. Search N -> U -> S
```
python nus_main.py search --pull-max-mm 25                       one run
python nus_main.py search --pull-max-mm 25 --trials 5 --record   5 runs + video for the report
```
For each letter: **0. measure the cables** where the arm is now: each cable is pulled 3 mm and
let back (the tip wiggles a little), printed as `cables here: M0 3.4 mm/mm at +88 deg, +0.7 up | ...`.
The calibration's model is only right near the rest pose, so this is what steers the tip.
**1. raise** the tip by **bending** the arm up (pulling all cables together would only stiffen
it), using only cables that also move the tip **toward** the letter (within ~60 deg), until it is
`--raise-above-mm` (default 20) above the tallest letter/obstacle, or as high as it goes that way,
or a motor's current rises by `--raise-max-ma` (default 300 mA). If no such cable raises it, it
doesn't raise ("no cable raises the tip while moving it toward the letter"). **2. move** to the
letter; staying high is only a weak extra goal, the direction to the letter comes first.
**3. come down** onto it (if it was raised) until it rests on it, then correct x/z.
**4.** If it stopped short, it measures the cables again there and tries once more.
`--no-raise` = skip step 1.
`--pull-max-mm` = most each cable may pull (calibration default 15 mm, which only reaches about
12-40 mm from the rest pose; 25 mm reaches about 20-65 mm and bends the arm further up). Max 40.
Ready screen: each letter shows "looks reachable" or "OUT OF REACH": move letters closer if
needed, then press **SPACE** (f = start with the letters found, q = quit).
Success = tip within 8 mm (horizontally) of the letter's magnet (`--success-mm` to change).
Route (`--route`, default `contact`): the tip goes **around** each obstacle's real outline (its
footprint polygon, not a circle), planned as the shortest path round them. `contact` keeps the
tip only `--hug-mm` (default 8, about the arm's radius) from obstacles, so the arm lies on them and
bends round them (tendons alone can't give the second bend), and `--clearance-mm` (default 15)
from the other letters. `--route avoid` keeps `--clearance-mm` from everything (no contact),
`--route straight` doesn't plan. Thick orange = footprint, thin orange = offset the tip keeps,
cyan = planned route. The ready screen lists the obstacles found; an obstacle only counts if it's
seen in 3 of the last 5 frames. A letter sitting on a block is fine: the block counts as an
obstacle for the other letters, and is ignored when going to that letter.
Results: `runs/search_results.csv` (one row per letter + one "ALL" row per trial, incl.
raised_mm and above_letter_mm = tip height above the letter at the end) and `runs/search_*.mp4`.

## 5. Retrieve: pick the letters up and put them in the holder (trial and error)
Teach the holder once per set-up (camera only): put the holder in place, then
```
python nus_main.py holder
```
click the middle of each slot (where that letter's magnet ends up) and press n / u / s; q saves.
Check the plan without moving (camera only), then run it:
```
python nus_main.py retrieve --dry-run                one look at the plan (30 s, then it stops)
python nus_main.py retrieve --record                 the real thing, with a video in runs/
python nus_main.py retrieve --no-place               holder not ready: only reach each magnet
```
What it does, until all three letters are in their slots (q in the window or Ctrl+C stops it):
1. scans the letters (their magnets) and the obstacles;
2. plans, for every letter, one route per side of the block it lies on (8 sides): around the
   obstacles (grown by the magnet tip's length, `tip_length_mm` = 8 in nus_config.json, or
   `--tip-mm`) to just outside that side, then straight in. Sides away from the middle are
   preferred (the tip is higher there). Cyan = route it takes, grey = the alternatives,
   coloured dots = approach points. The cheapest letter goes first;
3. drives it; if the tip doesn't move for `--stuck-s` 3 s (caught), or doesn't get there, the
   cables are let go (tip back to the middle), that side is marked failed, and it rescans and
   tries the next route (up to `--max-attempts` 5 per letter, then one more round of all sides);
4. at the magnet: lines up to 4 mm, touches it, backs off and checks the letter came along
   (it's gone from where it lay); carries it to its slot the same way, puts it down, goes back to
   the middle and checks the letter is in the slot (`--slot-tol-mm` 15).
Every attempt is logged in `runs/retrieve_log.csv` (letter, side, stage, result, reason).

## If something goes wrong
| Message | Do this |
|---|---|
| Could not open the camera / connect to the motors | Close EMIO Labs and other scripts; replug USB; check power |
| A cable spool has moved since calibration | Run step 2 again |
| Motor j turned but the tip never moved | Cable j is slack or off its pulley: check it, then step 1 |
| Calibration: "cable j barely moves the tip" | That cable isn't used for steering: check it's taut, then calibrate again |
| Raising stops with "cables getting tight" | The arm is bent as far as is safe; if it's clearly not strained, raise `--raise-max-ma` a little (watch the arm) |
| Raising takes the tip out of view ("lost the tip") | Lower `--raise-above-mm`, or move the camera so the raised tip stays in view |
| Many letters "out of reach" after 0 steps | Steering limit reached: use `--pull-max-mm 25` |
| Motor j did not follow its commands | Motor power / wiring |
| The tip is far off-centre | Arm not hanging freely, or something green near the robot |
| Letter "not found" / OUT OF REACH | Move it closer to the centre (inside the circle), teach it (step 3) |
| Letters detected in the wrong place | `stage_y_mm` in `nus_config.json` must match the surface height (-290 bare platform, about -228 raised) |
| `retrieve`: "No route left to try" | Every side failed twice: move the letter or the obstacle, or try `--route avoid` |
| `retrieve`: tip keeps getting caught on a block's edge | Raise `--tip-mm` (more clearance), or lower that block |
| `retrieve`: "letter didn't come along" | The magnet didn't take it: check the tip's magnet; `--success-mm 5` lines up closer |
| An obstacle isn't outlined | Click on it in `detect`: if it prints less than 8 mm above the surface, lower `obstacle_height_mm`; "no depth reading" = shiny/black top, put tape on it |
| Outlines on the platform / noise | Raise `obstacle_height_mm` (e.g. 12) |
| Part of the arm is outlined | It must touch the tip or reach up to the robot to be left out: tell me, or move the arm clear when starting |

Note: the tip hangs ~70 mm above letters on the bare platform, so "reached" means it is
directly **above** the letter. Picking up needs the platform raised (blue elevators).
