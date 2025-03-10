import mujoco
import mujoco.viewer
import numpy as np
import sys
import pdb
import time
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
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True #freejoints are showed as boxes that mask the hinge ones, so comment the former in the xml file for debug
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False # Show contact points
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False # Show contact forces
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True # Show collision bounding boxes
        while viewer.is_running():

            for i in range(data.ncon):
                contact = data.contact[i]
                print(f"Contact {i}:")
                print(f"  Normal    : {contact.frame[6:9]}")  # Contact normal
                print(f"  Tangent 1 : {contact.frame[0:3]} (Friction 'a')")
                print(f"  Tangent 2 : {contact.frame[3:6]} (Friction 'b')")


            mujoco.mj_step(model, data)
            viewer.sync()
            #time.sleep(0.01)
            #pdb.set_trace()
    
