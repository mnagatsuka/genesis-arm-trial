import genesis as gs

gs.init(backend=gs.cpu)  # start with CPU to keep it simple

scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

scene.build()

for _ in range(1000):
    scene.step()
