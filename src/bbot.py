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


sys.path.append("..")
import policies
import utils
from utils import plot_vectors
from assets import OmniWheelRef



# Load the model
model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)

#apply random force at init
random_force = np.random.uniform(-10, 10, size=3)
data.xfrc_applied[model.body("base").id, :3] = random_force

#print masses
total_mass = 0
for b_i in range(model.nbody):
    body_name = model.body(b_i).name
    cur_mass = model.body(body_name).mass.item()
    print(f"{body_name}: {cur_mass}")
    total_mass += cur_mass
print(colored(f"total_mass: {total_mass}", "magenta", attrs=["bold"]))

omni=OmniWheelRef()
ctrl_hist=[]

if 1:
    with mujoco.viewer.launch_passive(model, data) as viewer:

        fig_ax = None
        step_counter = 0
        plt.ion()
        while viewer.is_running():

            if data.time==0.0:
                step_counter=0

                k_vals=[200,50,20]

                #pid=policies.PID(dt=model.opt.timestep,k_d=-100,k_i=-100,k_p=-100)
                pid=policies.PID(dt=model.opt.timestep,
                        k_p=k_vals[0],
                        k_i=k_vals[1],
                        k_d=k_vals[2])


            imu_accel = data.sensordata[:3]  # First 3 values: Accelerometer
            imu_gyro = data.sensordata[3:6]  # Next 3 values: Gyroscope

            #print("Accelerometer:", imu_accel)
            #print("Gyroscope:", imu_gyro)

            body_id = model.body("base").id  
            position = data.xpos[body_id]  
            orientation = quaternion.quaternion(*data.xquat[body_id])
  
            with torch.no_grad():

                R_mat=quaternion.as_rotation_matrix(orientation)#this seems to be correct, not need for transpose
                ctrl_b,angle_deg, up_axis_global=pid.act(torch.tensor(R_mat).float())


                ctrl_c=torch.zeros(3)
                ctrl_c[0]=ctrl_b[1]*np.cos(utils.deg2rad(0))  + ctrl_b[0]*np.sin(utils.deg2rad(0))
                ctrl_c[1]=ctrl_b[1]*np.cos(utils.deg2rad(120))+ ctrl_b[0]*np.sin(utils.deg2rad(120))
                ctrl_c[2]=ctrl_b[1]*np.cos(utils.deg2rad(240))+ ctrl_b[0]*np.sin(utils.deg2rad(240))
                ctrl_c=torch.clamp(ctrl_c,min=-10,max=10)

                print("ctrl_wheels==",ctrl_c)
                print(colored("===========================","magenta",attrs=["bold"]))
                #ctrl_c_mine=omni.body_plane_to_control_space(ctrl_b.cpu().detach().numpy().reshape(2,1)).reshape(3)
                #ctrl_c=ctrl_c_mine

                ctrl_hist.append(ctrl_c.reshape(1,3))


                if angle_deg>20:
                    print(f"fail after {step_counter}")
                    #pdb.set_trace()
                    #plt.plot(pid.err_hist,label="err")
                    
                    mm=np.concatenate(pid.err_hist,0)
                    #plt.plot(mm[:,0],label="err0")
                    #plt.plot(mm[:,1],label="err1")
                    
                    #plt.plot(pid.integral_hist,label="int")
                    #plt.plot(pid.derivative_hist,label="der")
                   
                    #plt.title(f"fail after {step_counter}")
                    nn=np.concatenate(ctrl_hist,0)
                    for c_i in range(3):
                        plt.plot(nn[:,c_i],label=f"ctrl_{c_i}")
                    plt.legend(fontsize=15);plt.show()
                    pdb.set_trace()
            
                
            #pdb.set_trace()
           
            #fig_ax = plot_vectors(origins=np.array([[0, 0, 0]]),
            #                              directions=np.eye(3),
            #                              colors=["r", "g", "b"],
            #                              fig_ax=fig_ax,
            #                              scale_factor=10,
            #                              clear=True,
            #                              dashed=True)
            #fig_ax = plot_vectors(origins=np.array([[0, 0, 0]]),
            #        directions=ctrl_global.numpy().reshape(1,3),
            #            colors=["k"],
            #            fig_ax=fig_ax,
            #            scale_factor=10,
            #            clear=False)

            #fig_ax = plot_vectors(origins=np.array([[0, 0, 0]]),
            #        directions=up_axis_global.numpy().reshape(1,3),
            #            colors=["m"],
            #            fig_ax=fig_ax,
            #            scale_factor=10,
            #            clear=False)

            #if step_counter>3000:
            #    pdb.set_trace()

            #print("ctrl==",ctrl_c)
            data.ctrl[:] = - ctrl_c
            #print("ctrl_vec_norm==",np.linalg.norm(ctrl_c))

            mujoco.mj_step(model, data)
            step_counter += 1
            viewer.sync()
