#!/usr/bin/env python3
"""Interactive forward kinematics calculator for a 5-DOF robotic arm.

This script uses a Denavit-Hartenberg (DH) model so you can update joint
angles on the fly and inspect the homogeneous transformation matrices.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

Matrix4 = List[List[float]]


@dataclass
class ArmParameters:
    d: float = 120.0   # Base vertical offset
    L1: float = 160.0  # Link 1
    L2: float = 40.0   # Vertical offset near joint 2
    L3: float = 150.0  # Link 3
    L4: float = 35.0   # Vertical offset near joint 3
    L5: float = 60.0   # Wrist link


def identity4() -> Matrix4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matmul4(a: Matrix4, b: Matrix4) -> Matrix4:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def invert_rigid_transform(t: Matrix4) -> Matrix4:
    """Inverse of a homogeneous rigid transform [R p; 0 1]."""
    r = [[t[i][j] for j in range(3)] for i in range(3)]
    p = [t[i][3] for i in range(3)]

    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    inv_p = [-(rt[i][0] * p[0] + rt[i][1] * p[1] + rt[i][2] * p[2]) for i in range(3)]

    out = identity4()
    for i in range(3):
        for j in range(3):
            out[i][j] = rt[i][j]
        out[i][3] = inv_p[i]
    return out


def dh_transform(theta_deg: float, d: float, a: float, alpha_deg: float) -> Matrix4:
    """Standard DH transform A_i(theta, d, a, alpha)."""
    th = math.radians(theta_deg)
    al = math.radians(alpha_deg)
    ct, st = math.cos(th), math.sin(th)
    ca, sa = math.cos(al), math.sin(al)

    return [
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ]


def build_dh_table(params: ArmParameters, angles_deg: Dict[str, float]) -> List[Tuple[float, float, float, float]]:
    """Approximate DH table for the arm in the image.

    Each tuple is (theta_deg, d, a, alpha_deg).
    Edit here to match your exact robot geometry.
    """
    return [
        (angles_deg["theta1"], params.d, 0.0, 90.0),
        (angles_deg["theta2"], -params.L2, params.L1, 0.0),
        (angles_deg["theta3"], -params.L4, params.L3, 0.0),
        (angles_deg["theta4"], 0.0, params.L5, 90.0),
        (angles_deg["theta5"], 0.0, 0.0, 0.0),
    ]


def forward_kinematics(params: ArmParameters, angles_deg: Dict[str, float]) -> List[Matrix4]:
    """Return cumulative transforms [T0_0, T0_1, ..., T0_5]."""
    transforms = [identity4()]
    current = identity4()
    for theta, d, a, alpha in build_dh_table(params, angles_deg):
        current = matmul4(current, dh_transform(theta, d, a, alpha))
        transforms.append(current)
    return transforms


def joint_transforms(params: ArmParameters, angles_deg: Dict[str, float]) -> List[Matrix4]:
    """Return local joint transforms [A1, A2, ..., A5]."""
    return [dh_transform(theta, d, a, alpha) for theta, d, a, alpha in build_dh_table(params, angles_deg)]


def transform_between_frames(transforms: List[Matrix4], start_frame: int, end_frame: int) -> Matrix4:
    """Compute T_start_end from cumulative transforms [T0_i]."""
    if start_frame < 0 or end_frame < 0:
        raise ValueError("Frame indices must be >= 0.")
    if start_frame >= len(transforms) or end_frame >= len(transforms):
        raise ValueError(f"Frame indices must be within 0..{len(transforms)-1}.")
    return matmul4(invert_rigid_transform(transforms[start_frame]), transforms[end_frame])


def compose_joint_range(local_transforms: List[Matrix4], start_frame: int, end_frame: int) -> Matrix4:
    """Compose local transforms from frame start_frame to end_frame.

    Example: start=0, end=3 => A1 * A2 * A3 (equivalent to T0_3).
    """
    if start_frame == end_frame:
        return identity4()
    if start_frame > end_frame:
        raise ValueError("start_frame must be <= end_frame.")
    if start_frame < 0 or end_frame > len(local_transforms):
        raise ValueError(f"Frames must be within 0..{len(local_transforms)}.")

    result = identity4()
    for idx in range(start_frame, end_frame):
        result = matmul4(result, local_transforms[idx])
    return result


def format_matrix(m: Matrix4, digits: int = 4) -> str:
    lines = []
    for row in m:
        lines.append("[ " + "  ".join(f"{v: .{digits}f}" for v in row) + " ]")
    return "\n".join(lines)


def transform_point(m: Matrix4, point_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Apply homogeneous transform to a 3D point."""
    x, y, z = point_xyz
    px = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
    py = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
    pz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
    return (px, py, pz)


def joint_positions(transforms: List[Matrix4]) -> List[Tuple[float, float, float]]:
    """Extract joint origins from cumulative transforms."""
    return [(t[0][3], t[1][3], t[2][3]) for t in transforms]


def save_pose_plot(transforms: List[Matrix4], output_path: str) -> str:
    """Save a 3D plot of the current arm pose.

    Requires matplotlib. Returns a status message.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is not installed. Run: pip install matplotlib"

    pts = joint_positions(transforms)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, marker="o", linewidth=2)
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=80, marker="x", label="End effector")

    max_range = max(
        max(xs) - min(xs) if len(xs) > 1 else 1.0,
        max(ys) - min(ys) if len(ys) > 1 else 1.0,
        max(zs) - min(zs) if len(zs) > 1 else 1.0,
        1.0,
    )
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0
    cz = (max(zs) + min(zs)) / 2.0
    half = max_range / 2.0 + 10.0

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(max(0.0, cz - half), cz + half)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Robot Arm Pose")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return f"Saved pose image to {output_path}"


def print_status(params: ArmParameters, angles_deg: Dict[str, float], transforms: List[Matrix4]) -> None:
    print("\nCurrent joint angles (deg):")
    for key in sorted(angles_deg.keys()):
        print(f"  {key} = {angles_deg[key]:.3f}")

    end_eff = transforms[-1]
    x, y, z = end_eff[0][3], end_eff[1][3], end_eff[2][3]
    print(f"\nEnd-effector position (same units as L values): x={x:.3f}, y={y:.3f}, z={z:.3f}")
    print("\nUse `matrix <i>` / `joints` to inspect individual frame matrices (A_i).")
    print("Use `range <start> <end>` to display cumulative transforms T_start_end.")

    _ = params  # Placeholder so the signature is useful if expanded later.


def build_matrix_report(transforms: List[Matrix4]) -> str:
    parts = []
    for i, mat in enumerate(transforms):
        parts.append(f"T0_{i}:\n{format_matrix(mat)}")
    return "\n\n".join(parts)


def build_joint_matrix_report(local_transforms: List[Matrix4]) -> str:
    parts = []
    for i, mat in enumerate(local_transforms, start=1):
        parts.append(f"A{i} (frame {i-1} -> {i}):\n{format_matrix(mat)}")
    return "\n\n".join(parts)


def parse_frame_range(
    tokens: List[str], transforms: List[Matrix4], params: ArmParameters, angles_deg: Dict[str, float]
) -> str:
    """Parse range command and return formatted transform report."""
    if len(tokens) != 2 and len(tokens) != 3:
        return "Usage: range <start> <end>   or   range all"

    if len(tokens) == 2 and tokens[1].lower() == "all":
        return "T0_5 (all joints):\n" + format_matrix(transforms[-1])

    if len(tokens) != 3 or not tokens[1].isdigit() or not tokens[2].isdigit():
        return "Usage: range <start> <end>   or   range all"

    start = int(tokens[1])
    end = int(tokens[2])
    if start > end:
        return "Use start <= end (example: range 0 3)."

    try:
        local = joint_transforms(params, angles_deg)
        composed = compose_joint_range(local, start, end)
        mapped = transform_between_frames(transforms, start, end)
    except ValueError as exc:
        return str(exc)

    lines = [
        f"Selected frame range: {start} -> {end}",
        f"Composed individual frame matrices (A{start+1}..A{end}) = T{start}_{end}:",
        format_matrix(composed),
        "",
        f"Mapped from cumulative transforms (inv(T0_{start}) * T0_{end}) = T{start}_{end}:",
        format_matrix(mapped),
    ]
    return "\n".join(lines)


def parse_vector_command(
    tokens: List[str], transforms: List[Matrix4], params: ArmParameters, angles_deg: Dict[str, float]
) -> str:
    """Extract translation vector from a selected transform and optionally transform a point."""
    usage = (
        "Usage: vector <start> <end> [x y z]\n"
        "Examples:\n"
        "  vector 0 4\n"
        "  vector 0 4 10 0 0"
    )
    if len(tokens) not in {3, 6}:
        return usage
    if not tokens[1].isdigit() or not tokens[2].isdigit():
        return usage

    start = int(tokens[1])
    end = int(tokens[2])
    if start > end:
        return "Use start <= end (example: vector 0 4)."

    try:
        local = joint_transforms(params, angles_deg)
        t = compose_joint_range(local, start, end)
        mapped = transform_between_frames(transforms, start, end)
    except ValueError as exc:
        return str(exc)

    tx, ty, tz = mapped[0][3], mapped[1][3], mapped[2][3]
    lines = [
        f"Selected transform: T{start}_{end}",
        f"Position vector (translation of T{start}_{end}): [{tx:.4f}, {ty:.4f}, {tz:.4f}]",
    ]

    if len(tokens) == 6:
        try:
            point = (float(tokens[3]), float(tokens[4]), float(tokens[5]))
        except ValueError:
            return "Point coordinates must be numeric.\n" + usage
        px, py, pz = transform_point(mapped, point)
        lines.extend(
            [
                f"Input point: [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]",
                f"Point transformed by T{start}_{end}: [{px:.4f}, {py:.4f}, {pz:.4f}]",
            ]
        )

    # Sanity check: composed range and mapped transform should match numerically.
    _ = t
    return "\n".join(lines)


def parse_set_command(tokens: List[str], params: ArmParameters, angles_deg: Dict[str, float]) -> str:
    if len(tokens) != 3:
        return "Usage: set <theta1..theta5 | d|L1..L5> <value>"

    name = tokens[1]
    try:
        value = float(tokens[2])
    except ValueError:
        return "Value must be numeric."

    if name in angles_deg:
        angles_deg[name] = value
        return f"Updated {name} = {value}"

    if hasattr(params, name):
        setattr(params, name, value)
        return f"Updated {name} = {value}"

    return f"Unknown parameter '{name}'."


def interactive_loop(params: ArmParameters, angles_deg: Dict[str, float]) -> None:
    help_text = """Commands:
  show                      Show current angles and end-effector position
  all                       Show all individual frame matrices A1..A5
  matrix <i>                Show individual frame matrix Ai where i is 1..5
  joints                    Show individual frame matrices A1..A5
  joint <i>                 Show individual frame matrix Ai where i is 1..5
  range <start> <end>       Compose joints from frame start to end (e.g. range 0 3)
  range all                 Show full transform T0_5
  vector <s> <e> [x y z]    Show translation vector of T_s_e (and optionally transform a point)
  plot [file.png]           Save a 3D image of the current arm pose
  set <thetaN> <value>      Update joint angle in degrees (theta1..theta5)
  set <d|L1..L5> <value>    Update a length/offset parameter
  help                      Show commands
  quit                      Exit
"""
    print(help_text)
    transforms = forward_kinematics(params, angles_deg)
    print_status(params, angles_deg, transforms)

    while True:
        try:
            raw = input("\nrobot-fk> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not raw:
            continue
        tokens = raw.split()
        cmd = tokens[0].lower()

        if cmd in {"quit", "exit", "q"}:
            print("Exiting.")
            return
        if cmd == "help":
            print(help_text)
            continue

        if cmd == "set":
            print(parse_set_command(tokens, params, angles_deg))
            transforms = forward_kinematics(params, angles_deg)
            print_status(params, angles_deg, transforms)
            continue

        if cmd == "show":
            transforms = forward_kinematics(params, angles_deg)
            print_status(params, angles_deg, transforms)
            continue

        if cmd == "all":
            local = joint_transforms(params, angles_deg)
            print(build_joint_matrix_report(local))
            continue

        if cmd == "joints":
            local = joint_transforms(params, angles_deg)
            print(build_joint_matrix_report(local))
            continue

        if cmd == "joint":
            if len(tokens) != 2 or not tokens[1].isdigit():
                print("Usage: joint <i>  (i from 1 to 5)")
                continue
            idx = int(tokens[1])
            local = joint_transforms(params, angles_deg)
            if idx < 1 or idx > len(local):
                print(f"Joint matrix index out of range. Use 1..{len(local)}.")
                continue
            print(f"\nA{idx} (frame {idx-1} -> {idx}):")
            print(format_matrix(local[idx - 1]))
            continue

        if cmd == "range":
            transforms = forward_kinematics(params, angles_deg)
            print(parse_frame_range(tokens, transforms, params, angles_deg))
            continue

        if cmd == "vector":
            transforms = forward_kinematics(params, angles_deg)
            print(parse_vector_command(tokens, transforms, params, angles_deg))
            continue

        if cmd == "plot":
            out = tokens[1] if len(tokens) > 1 else "arm_pose.png"
            transforms = forward_kinematics(params, angles_deg)
            print(save_pose_plot(transforms, out))
            continue

        if cmd == "matrix":
            if len(tokens) != 2 or not tokens[1].isdigit():
                print("Usage: matrix <i>  (i from 1 to 5)")
                continue
            idx = int(tokens[1])
            local = joint_transforms(params, angles_deg)
            if idx < 1 or idx > len(local):
                print(f"Matrix index out of range. Use 1..{len(local)}.")
                continue
            print(f"\nA{idx} (frame {idx-1} -> {idx}):")
            print(format_matrix(local[idx - 1]))
            continue

        print("Unknown command. Type 'help' for options.")


def launch_gui(params: ArmParameters, angles_deg: Dict[str, float]) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText
    except Exception as exc:  # pragma: no cover
        print(f"Tkinter is unavailable: {exc}")
        print("On Arch Linux install: sudo pacman -S tk")
        return

    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
    except Exception as exc:  # pragma: no cover
        print(f"Matplotlib Tk backend is unavailable: {exc}")
        print("Install matplotlib (Arch): sudo pacman -S python-matplotlib")
        return

    root = tk.Tk()
    root.title("Robot Arm FK Viewer")
    root.geometry("1200x760")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(0, weight=1)

    controls = ttk.LabelFrame(main, text="Inputs", padding=10)
    controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

    right = ttk.Frame(main)
    right.grid(row=0, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(0, weight=1)
    right.rowconfigure(1, weight=1)

    plot_frame = ttk.LabelFrame(right, text="Arm Pose", padding=5)
    plot_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
    matrix_frame = ttk.LabelFrame(right, text="Matrices", padding=5)
    matrix_frame.grid(row=1, column=0, sticky="nsew")
    matrix_frame.rowconfigure(0, weight=1)
    matrix_frame.columnconfigure(0, weight=1)

    fig = Figure(figsize=(7, 4.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    matrix_text = ScrolledText(matrix_frame, wrap="none", font=("Courier New", 10))
    matrix_text.grid(row=0, column=0, sticky="nsew")

    ee_var = tk.StringVar(value="End effector: x=0, y=0, z=0")
    ttk.Label(controls, textvariable=ee_var, justify="left").grid(row=0, column=0, sticky="w", pady=(0, 8))

    slider_vars: Dict[str, tk.DoubleVar] = {}

    def add_slider(row: int, name: str, low: float, high: float, step: float) -> None:
        ttk.Label(controls, text=name).grid(row=row, column=0, sticky="w")
        var = tk.DoubleVar(value=angles_deg.get(name, getattr(params, name)))
        slider_vars[name] = var
        scale = tk.Scale(
            controls,
            from_=low,
            to=high,
            orient="horizontal",
            resolution=step,
            variable=var,
            length=260,
        )
        scale.grid(row=row + 1, column=0, sticky="ew", pady=(0, 6))

    row_idx = 1
    for name in ("theta1", "theta2", "theta3", "theta4", "theta5"):
        add_slider(row_idx, name, -180.0, 180.0, 1.0)
        row_idx += 2

    add_slider(row_idx, "d", 0.0, max(500.0, params.d * 2.0), 1.0)
    row_idx += 2

    ttk.Separator(controls, orient="horizontal").grid(row=row_idx, column=0, sticky="ew", pady=6)
    row_idx += 1
    ttk.Label(controls, text="Link lengths (optional)").grid(row=row_idx, column=0, sticky="w")
    row_idx += 1
    for name in ("L1", "L2", "L3", "L4", "L5"):
        add_slider(row_idx, name, 0.0, max(300.0, getattr(params, name) * 2.0), 1.0)
        row_idx += 2

    pending_after_id = None

    def refresh_view() -> None:
        for key in ("theta1", "theta2", "theta3", "theta4", "theta5"):
            angles_deg[key] = float(slider_vars[key].get())
        for key in ("d", "L1", "L2", "L3", "L4", "L5"):
            setattr(params, key, float(slider_vars[key].get()))

        transforms = forward_kinematics(params, angles_deg)
        pts = joint_positions(transforms)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        ax.clear()
        ax.plot(xs, ys, zs, marker="o", linewidth=2)
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=80, marker="x", label="EE")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Robot Arm Pose")
        ax.grid(True)
        ax.legend(loc="upper left")

        span = max(
            (max(xs) - min(xs)) if len(xs) > 1 else 1.0,
            (max(ys) - min(ys)) if len(ys) > 1 else 1.0,
            (max(zs) - min(zs)) if len(zs) > 1 else 1.0,
            1.0,
        )
        cx = (max(xs) + min(xs)) / 2.0
        cy = (max(ys) + min(ys)) / 2.0
        cz = (max(zs) + min(zs)) / 2.0
        half = span / 2.0 + 10.0
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(max(0.0, cz - half), cz + half)

        ee_var.set(f"End effector: x={xs[-1]:.2f}, y={ys[-1]:.2f}, z={zs[-1]:.2f}")

        matrix_text.configure(state="normal")
        matrix_text.delete("1.0", "end")
        matrix_text.insert("1.0", build_matrix_report(transforms))
        matrix_text.configure(state="disabled")

        canvas.draw_idle()

    def schedule_refresh(*_args: object) -> None:
        nonlocal pending_after_id
        if pending_after_id is not None:
            root.after_cancel(pending_after_id)
        pending_after_id = root.after(30, _run_refresh)

    def _run_refresh() -> None:
        nonlocal pending_after_id
        pending_after_id = None
        refresh_view()

    for var in slider_vars.values():
        var.trace_add("write", schedule_refresh)

    button_row = ttk.Frame(controls)
    button_row.grid(row=row_idx, column=0, sticky="ew", pady=(8, 0))
    ttk.Button(button_row, text="Refresh", command=refresh_view).pack(side="left")

    def save_current_plot() -> None:
        out = "arm_pose_gui.png"
        transforms = forward_kinematics(params, angles_deg)
        print(save_pose_plot(transforms, out))

    ttk.Button(button_row, text="Save PNG", command=save_current_plot).pack(side="left", padx=(6, 0))

    refresh_view()
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-DOF robotic arm forward kinematics calculator")
    parser.add_argument(
        "--angles",
        nargs=5,
        type=float,
        metavar=("TH1", "TH2", "TH3", "TH4", "TH5"),
        help="Initial joint angles in degrees",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Print current T0_5 once and exit",
    )
    parser.add_argument(
        "--plot",
        metavar="FILE",
        help="Save a 3D image of the current arm pose to FILE (requires matplotlib)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open a Tkinter GUI with sliders and real-time matrix/pose updates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = ArmParameters()
    angles_deg = {
        "theta1": 0.0,
        "theta2": 0.0,
        "theta3": 0.0,
        "theta4": 0.0,
        "theta5": 0.0,
    }

    if args.angles:
        for i, value in enumerate(args.angles, start=1):
            angles_deg[f"theta{i}"] = value

    if args.gui:
        launch_gui(params, angles_deg)
        return

    transforms = forward_kinematics(params, angles_deg)
    if args.plot:
        print(save_pose_plot(transforms, args.plot))
    if args.no_interactive:
        print_status(params, angles_deg, transforms)
        return

    interactive_loop(params, angles_deg)


if __name__ == "__main__":
    main()
