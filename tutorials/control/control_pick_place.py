import numpy as np
import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

# small cube in front of the arm
cube = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.45, 0.0, 0.02))
)

scene.build()

# End-effector link name comes from the MJCF asset
ee_link = franka.get_link("hand")

# Helper: move end-effector using IK
def move_ee_to(target_pos, target_quat=None, steps=200):
    # Solve IK once, then smoothly track the target with position control
    qpos_target = franka.inverse_kinematics(
        link=ee_link,
        pos=np.array(target_pos, dtype=np.float32),
        quat=None if target_quat is None else np.array(target_quat, dtype=np.float32),
    )
    qpos_start = franka.get_dofs_position()
    for i in range(steps):
        alpha = (i + 1) / steps
        qpos_step = qpos_start + alpha * (qpos_target - qpos_start)
        franka.control_dofs_position(qpos_step)
        scene.step()

# 1) move above cube
move_ee_to(target_pos=(0.45, 0.0, 0.20), steps=200)
# 2) go down
move_ee_to(target_pos=(0.45, 0.0, 0.06), steps=200)

# 3) "suction": weld cube to gripper (imitate grasp)
scene.rigid_solver.add_weld_constraint(ee_link.idx, cube.base_link_idx)

# 4) lift + move
move_ee_to(target_pos=(0.45, 0.0, 0.25), steps=200)
move_ee_to(target_pos=(0.30, 0.20, 0.25), steps=250)

# 5) release
scene.rigid_solver.delete_weld_constraint(ee_link.idx, cube.base_link_idx)

for _ in range(300):
    scene.step()
