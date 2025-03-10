import mujoco
import mujoco.viewer
import numpy as np
import sys
import pdb
from termcolor import colored

# Load the model
model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)
#model.opt.gravity = (0, 0, 0)

#print masses 
total_mass=0
for b_i in range(model.nbody):
    body_name=model.body(b_i).name
    cur_mass=model.body(body_name).mass.item()
    print(f"{body_name}: {cur_mass}")
    total_mass+=cur_mass
print(colored(f"total_mass: {total_mass}","magenta",attrs=["bold"]))



if 1:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        #viewer.cam.distance = 500.0
        #viewer.cam.azimuth = 90
        #viewer.cam.elevation = -45
        #viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True  #freejoints are showed as boxes that mask the hinge ones, so comment the former in the xml file for debug
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            #pdb.set_trace()
    
