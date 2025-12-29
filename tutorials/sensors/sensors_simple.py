import numpy as np
import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)

franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
end_effector = franka.get_link('hand')

imu = scene.add_sensor(
    gs.sensors.IMU(
        entity_idx=franka.idx,
        link_idx_local=end_effector.idx_local,
        pos_offset=(0.0, 0.0, 0.15),
        acc_noise=(0.01, 0.01, 0.01),
        gyro_noise=(0.01, 0.01, 0.01),
        draw_debug=True,
    )
)

scene.build()

for i in range(300):
    scene.step()

    # センサーデータ読み取り
    data = imu.read()
    ground_truth = imu.read_ground_truth()

    print(f"加速度: {data.lin_acc}")
    print(f"角速度: {data.ang_vel}")