# Robot Arm Forward Kinematics (Python)

This project provides a Python script (`robot_arm_fk.py`) to calculate the forward kinematics matrices of a robotic arm as joint angles change.

It supports:
- Interactive angle updates (`theta1..theta5`)
- Linear offset update (`d`)
- Printing homogeneous transformation matrices (`T0_i`)
- Saving a simple 3D image of the current arm pose (optional, with `matplotlib`)

## What the script calculates

The script uses a Denavit-Hartenberg (DH) model to compute cumulative transforms:
- `T0_0` (identity)
- `T0_1`
- `T0_2`
- `T0_3`
- `T0_4`
- `T0_5` (end-effector pose)

The current arm geometry is an approximation based on your image. If your physical arm has different axis directions or offsets, edit `build_dh_table()` in `robot_arm_fk.py`.

## Requirements

- Python 3.9+ (3.8 also likely works)
- Optional for pose image export: `matplotlib`

## Setup

1. Open a terminal in this project folder:
```bash
cd /home/KQ/Documents/Robotica
```

2. (Optional but recommended) Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install optional plotting dependency (only needed for image export):
```bash
pip install matplotlib
```

## Run the program

Interactive mode:
```bash
python3 robot_arm_fk.py
```

One-shot run with custom angles:
```bash
python3 robot_arm_fk.py --no-interactive --angles 10 20 30 40 50
```

One-shot run + save pose image:
```bash
python3 robot_arm_fk.py --no-interactive --angles 10 20 30 40 50 --plot arm_pose.png
```

## Interactive commands

After starting `python3 robot_arm_fk.py`, you can use:

- `show` -> show current joint angles, end-effector position, and `T0_5`
- `all` -> print all cumulative transforms `T0_i`
- `matrix <i>` -> print `T0_i` (where `i` is `0..5`)
- `set theta1 30` -> update a joint angle (degrees)
- `set theta2 -45`
- `set d 180` -> update the linear movement/offset `d`
- `set L1 200` -> update a link length parameter
- `plot` -> save current pose image to `arm_pose.png`
- `plot my_pose.png` -> save image with a custom filename
- `help`
- `quit`

## About `d` (linear movement)

Yes, `d` is included. In the current DH model:
- `d` is the first joint/frame offset along the base axis (treated like a prismatic/linear input)
- You can change it live with:

```text
set d 150
```

## Can we show an image of the current arm position?

Yes. The script now supports saving a simple 3D pose image:

- In interactive mode: `plot` or `plot filename.png`
- In one-shot mode: `--plot filename.png`

Notes:
- This is a kinematic visualization (line + joint points), not a full CAD-rendered model.
- It uses the joint positions derived from the current transformation matrices.

## Customizing the robot model

If the matrices do not match your real robot, update:

- `ArmParameters` in `robot_arm_fk.py` for lengths/offsets
- `build_dh_table()` in `robot_arm_fk.py` for DH frame definitions (`theta, d, a, alpha`)

That is the main place where your robot's exact geometry is encoded.
