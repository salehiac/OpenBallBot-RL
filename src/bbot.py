import numpy as np
import sys
import pdb
import time
import matplotlib.pyplot as plt
import quaternion

import mujoco
import mujoco.viewer
import torch
from termcolor import colored


import policies
from utils import plot_vectors



# Load the model
model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)
#model.opt.gravity = (0, 0, 0)

#print masses
total_mass = 0
for b_i in range(model.nbody):
    body_name = model.body(b_i).name
    cur_mass = model.body(body_name).mass.item()
    print(f"{body_name}: {cur_mass}")
    total_mass += cur_mass
print(colored(f"total_mass: {total_mass}", "magenta", attrs=["bold"]))

pid=policies.PID()

if 1:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[
            mujoco.mjtVisFlag.
            mjVIS_JOINT] = False #freejoints are showed as boxes that mask the hinge ones, so comment the former in the xml file for debug
        viewer.opt.flags[mujoco.mjtVisFlag.
                         mjVIS_CONTACTPOINT] = False  # Show contact points
        viewer.opt.flags[mujoco.mjtVisFlag.
                         mjVIS_CONTACTFORCE] = False  # Show contact forces
        viewer.opt.flags[
            mujoco.mjtVisFlag.
            mjVIS_CONVEXHULL] = False # Show collision bounding boxes

        fig_ax = None
        step_counter = 0
        plt.ion()
        while viewer.is_running():

            if 0:
                for ii in range(data.ncon):
                    contact = data.contact[ii]
                    contact_point = contact.pos
                    tangent_1 = contact.frame[0:3]*contact.friction[0]
                    tangent_2 = contact.frame[3:6]*contact.friction[1]
                    normal = contact.frame[6:9]

                    #first plot axes
                    fig_ax = plot_vectors(origins=np.array([[0, 0, 0]]),
                                          directions=np.eye(3),
                                          colors=["r", "g", "b"],
                                          fig_ax=fig_ax,
                                          scale_factor=10,
                                          clear=True,
                                          dashed=True)

                    #now plot the tangential contact forces
                    origins = np.array(contact_point).reshape(
                        1, -1)  #all three vectors have the same origin
                    directions = np.concatenate(
                        [tangent_1.reshape(1, -1),
                         tangent_2.reshape(1, -1)], 0)
                    fig_ax = plot_vectors(origins=origins,
                                          directions=directions,
                                          colors=["k", "m"],
                                          fig_ax=fig_ax,
                                          scale_factor=10,
                                          clear=False)
                    np.set_printoptions(precision=3,suppress=True)
                    print(contact.friction)
                    #print(step_counter)

            imu_accel = data.sensordata[:3]  # First 3 values: Accelerometer
            imu_gyro = data.sensordata[3:6]  # Next 3 values: Gyroscope

            print("Accelerometer:", imu_accel)
            print("Gyroscope:", imu_gyro)

            body_id = model.body("base").id  
            position = data.xpos[body_id]  
            orientation = quaternion.quaternion(*data.xquat[body_id])
            print("position:", position)
            #print("orientation:", orientation)

            with torch.no_grad():

                R_mat=quaternion.as_rotation_matrix(orientation)
                ctrl,err_norm=pid.act(torch.tensor(R_mat).float())
                print(colored(err_norm,"red",attrs=["bold"]))
                ctrl=ctrl.detach().cpu().numpy().reshape(3)
            print("ctrl==",ctrl)
            #data.ctrl[:] = ctrl

            mujoco.mj_step(model, data)
            step_counter += 1
            viewer.sync()
            time.sleep(0.01)
            #pdb.set_trace()
