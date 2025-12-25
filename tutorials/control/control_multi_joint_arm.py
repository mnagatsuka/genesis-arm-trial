import genesis as gs
import numpy as np
import threading

def main():
    # Initialize
    gs.init(backend=gs.cpu)

    # Create the scene
    scene = gs.Scene(
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (0, -3.5, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            res           = (960, 640),
            max_FPS       = 60,
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        show_viewer = True,
    )

    # Add entities to the scene
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file  = 'xml/franka_emika_panda/panda.xml',
        ),
    )

    # Build the scene
    scene.build()

    # Run the simulation in the main thread (macOS viewer requires this)
    run_sim(scene, franka)


def run_sim(scene, franka):
    # Map joint names to local DoF indices
    jnt_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
        'finger_joint1',
        'finger_joint2',
    ]
    dofs_idx = []
    for name in jnt_names:
        dofs_idx.extend(franka.get_joint(name).dofs_idx_local)

    # Set position gains for the entity's DoFs
    franka.set_dofs_kp(
        kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        dofs_idx_local = dofs_idx,
    )

    # Set velocity gains for the entity's DoFs
    franka.set_dofs_kv(
        kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        dofs_idx_local = dofs_idx,
    )

    # Set force range for the entity's DoFs (safety)
    franka.set_dofs_force_range(
        lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
        dofs_idx_local = dofs_idx,
    )

    # Hard reset
    for i in range(150):
        # Set the entity's DoF positions
        if i < 50:
            franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
        # Set the entity's DoF positions
        elif i < 100:
            franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
        # Set the entity's DoF positions
        else:
            franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)

        scene.step()

    # PD control
    for i in range(1250):
        if i == 0:
            # Set target positions for the entity's DoFs
            franka.control_dofs_position(
                np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 250:
            # Set target positions for the entity's DoFs
            franka.control_dofs_position(
                np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 500:
            # Set target positions for the entity's DoFs
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )
        elif i == 750:
            # Set target positions for the entity's DoFs
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
                dofs_idx[1:],
            )
            # Set target velocity for the entity's DoFs
            franka.control_dofs_velocity(
                np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
                dofs_idx[:1],
            )
        elif i == 1000:
            # Set target control force for the entity's DoFs
            franka.control_dofs_force(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )

        # Control force computed from the control command
        print('control force:', franka.get_dofs_control_force(dofs_idx))

        # Actual force applied to the joints
        print('internal force:', franka.get_dofs_force(dofs_idx))

        scene.step()


if __name__ == "__main__":
    main()
