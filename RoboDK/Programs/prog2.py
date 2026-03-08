from robodk.robolink import Robolink, ITEM_TYPE_ROBOT, ITEM_TYPE_FRAME, ITEM_TYPE_TOOL, ITEM_TYPE_TARGET
from robodk.robomath import transl, rotx, pi

RDK = Robolink()

station_path = r"/home/KQ/Documents/Robotica/RoboDK/Stations/ScaraExample.rdk"
st = RDK.AddFile(station_path)
print("Station load:", st.Name() if st.Valid() else "FAILED")

robot = RDK.ItemUserPick("Seleccione el robot", ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception("No se seleccionó un robot válido")
print("Robot:", robot.Name())

# Try to auto-pick a frame/tool if your station has exactly one of each (common)
frames = RDK.ItemList(ITEM_TYPE_FRAME)
tools  = RDK.ItemList(ITEM_TYPE_TOOL)

if frames and frames[0].Valid():
    robot.setPoseFrame(frames[0])
    print("Using frame:", frames[0].Name())
else:
    print("No frames found; using robot default frame")

if tools and tools[0].Valid():
    robot.setPoseTool(tools[0])
    print("Using tool:", tools[0].Name())
else:
    print("No tools found; using robot default tool/TCP")

# Orientation: tool pointing down
R_down = rotx(pi)

# Test points (start closer to base to increase chance)
puntos = [
    (200, 0, 300),
    (300, 0, 300),
    (300, 100, 300),
    (200, 100, 300),
]

# Check reachability before moving
for i, (x, y, z) in enumerate(puntos, start=1):
    pose = transl(x, y, z) * R_down

    # Solve IK first (does not move)
    jnts = robot.SolveIK(pose)
    if jnts is None or len(jnts) == 0:
        print(f"Point {i} NOT reachable: {(x,y,z)}")
        continue

    # If reachable, move
    print(f"Point {i} reachable -> moving: {(x,y,z)}")
    robot.MoveJ(pose)

print("Done")