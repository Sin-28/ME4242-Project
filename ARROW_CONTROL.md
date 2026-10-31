# Arrow-key rod control

Requires the project's EMIO environment with `emioapi` and `dynamixel_sdk` installed.
Run commands below from this repository directory in a VS Code terminal.

Place the rod at its center position, then run this in the VS Code terminal:

```bash
python emio_arrow_control.py
```

The default direction mapping is up=0, down=2, left=1, right=3. Motors 1 and 3
pull in the increasing encoder direction by default. Each arrow
press changes the corresponding bend by 5° of motor rotation. Opposite arrows
return that axis toward center before pulling its opposite motor. Space returns
both axes to the startup position; Q or Esc exits and releases motor torque.
The controller switches the motors to Extended Position Mode, so a goal can cross
the 0–4095 single-turn pulse boundary. There is no configured angle limit;
use `--step 2` for smaller movements. A larger combined bend previously
triggered a motor overload, so watch the rod and cable loading closely. The
program stops if a motor stalls or reports a hardware fault.

The script assumes motors 0 and 2 pull when encoder pulses decrease, while
motors 1 and 3 pull when pulses increase. To override the default reverse
list, specify all applicable IDs, for example `--reverse 1 2 3`.

To record a manual session and replay it later:

```bash
python emio_arrow_control.py --record my_motion.json
python emio_arrow_control.py --play my_motion.json
```

To switch between manual control, recording, and playback without disconnecting
the motors, run `python emio_arrow_control.py --session`. Press `R` and type a
digit 0–9 to record to `motion_N.json` (for example, `R` then `2` writes
`motion_2.json`); press `R` again to save. Press `P` then a digit to play the
matching file; playback returns to manual control when finished. Press
`P` during playback to stop it, or `Q`/Esc to exit. The rod stays at its current
pose when modes switch. A session recording stores movements relative to the
pose when recording began, and session playback applies them from the current
pose using the motor configuration saved in the file. Existing `motion_1.json`
and `motion_2.json` can be selected with `P` then `1` or `2`.

Recording saves the timing and commanded bend after each successful key press,
including Space. Put the rod at its center position before each run. Playback
uses that run's starting motor positions as its center and stops on Q or Esc.
Playback uses the motor mapping and reverse directions saved in each recording,
so older recordings keep their original motor commands. New recordings use the
current defaults unless those options are overridden.
