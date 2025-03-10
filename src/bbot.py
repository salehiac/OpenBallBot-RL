import mujoco
import mujoco.viewer
import numpy as np
import sys
import pdb

# Load the model
model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)
model.opt.gravity = (0, 0, 0)
# Create a viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    #viewer.cam.distance = 500.0
    #viewer.cam.azimuth = 90
    #viewer.cam.elevation = -45
    #viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        #pdb.set_trace()

