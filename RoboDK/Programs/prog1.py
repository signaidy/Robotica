from robodk.robolink import Robolink, ITEM_TYPE_ROBOT
from robodk.robomath import transl, rotz
from math import pi

def mat_to_joint_list(jnts):
    """Convert RoboDK Mat/array result to a flat python list of floats."""
    if jnts is None:
        return None

    # RoboDK Mat type usually has .list()
    if hasattr(jnts, "list"):
        lst = jnts.list()
        # lst may be nested (e.g., [[j1, j2, j3, j4]])
        if len(lst) == 1 and isinstance(lst[0], list):
            return [float(x) for x in lst[0]]
        return [float(x) for x in lst]

    # numpy-like
    if hasattr(jnts, "flatten"):
        return [float(x) for x in jnts.flatten().tolist()]

    # already a list/tuple
    if isinstance(jnts, (list, tuple)):
        if len(jnts) == 1 and isinstance(jnts[0], (list, tuple)):
            return [float(x) for x in jnts[0]]
        return [float(x) for x in jnts]

    return None

RDK = Robolink()
robot = RDK.ItemUserPick("Seleccione el robot", ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception("No se seleccionó un robot válido")

print("Robot:", robot.Name())
print("Robot has", robot.Joints().size(1) if hasattr(robot.Joints(), "size") else "?", "joints (current)")

yaw = 0 * pi/180

# Try a single point first
x, y, z = (200, 0, 150)
pose = transl(x, y, z) * rotz(yaw)

jnts_raw = robot.SolveIK(pose)
jnts = mat_to_joint_list(jnts_raw)

print("SolveIK raw:", jnts_raw)
print("SolveIK joints list:", jnts)

if not jnts or len(jnts) < 3:
    raise Exception("IK did not return a full joint solution. Try using a Target from the station (see below).")

robot.MoveJ(jnts)
print("Moved OK")