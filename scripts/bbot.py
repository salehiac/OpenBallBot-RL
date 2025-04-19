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


class RGBDInputs:

    def __init__(self,mjc_model, cam_name, height, width, normalize=True):

        self.renderer_rgb=mujoco.Renderer(model, width=width, height=height)
        self.renderer_d=mujoco.Renderer(model, width=width, height=height)
        self.renderer_d.enable_depth_rendering()

        self.cam_name=cam_name
        self.normalize=normalize

    def __call__(self, data):

        self.renderer_rgb.update_scene(data, camera=self.cam_name)  
        self.renderer_d.update_scene(data, camera=self.cam_name)  
        rgb=self.renderer_rgb.render().astype("float")
        depth=np.expand_dims(self.renderer_d.render(),axis=-1)

        if self.normalize:
            rgb/=255.0
        return np.concatenate([rgb, depth],-1)


# Load the model
model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)

#apply random force at init
random_force = np.random.uniform(-15, 15, size=3)
random_force=np.array([5,5,5])
print(random_force)
data.xfrc_applied[model.body("base").id, :3] = random_force

#print masses
total_mass = 0
for b_i in range(model.nbody):
    body_name = model.body(b_i).name
    cur_mass = model.body(body_name).mass.item()
    print(f"{body_name}: {cur_mass}")
    total_mass += cur_mass
print(colored(f"total_mass: {total_mass}", "magenta", attrs=["bold"]))

ctrl_hist=[]

#NOTE: you can access the cameras through the GUI though. This is for getting rgbd images to feed to policies etc
rgbd_inputs=[RGBDInputs(model, cam_name="cam_0", height=480, width=480), RGBDInputs(model, cam_name="cam_1", height=480, width=480)]
#cam_fig, cam_fig_ax=plt.subplots(2,2)

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

            rgbd_0=rgbd_inputs[0](data)
            rgbd_1=rgbd_inputs[1](data)
            #cam_fig_ax[0,0].imshow(rgbd_0[:,:,:3])
            #cam_fig_ax[0,1].imshow(rgbd_0[:,:,3])
            #cam_fig_ax[1,0].imshow(rgbd_1[:,:,:3])
            #cam_fig_ax[1,1].imshow(rgbd_1[:,:,3])
            #plt.pause(0.0001)
            #plt.show()

            #print("Accelerometer:", imu_accel)
            #print("Gyroscope:", imu_gyro)


            body_id = model.body("base").id  
            position = data.xpos[body_id]  
            orientation = quaternion.quaternion(*data.xquat[body_id])
  
            with torch.no_grad():

                R_mat=quaternion.as_rotation_matrix(orientation)#this seems to be correct, not need for transpose
                ctrl_c,angle_deg=pid.act(torch.tensor(R_mat).float())


                print("ctrl_wheels==",ctrl_c,"step==",step_counter,"random_force==",random_force)
                print(colored("===========================","magenta",attrs=["bold"]))

                ctrl_hist.append(ctrl_c.reshape(1,3))


                if angle_deg>20:
                    print(f"fail after {step_counter}")
                    
                    mm=np.concatenate(pid.err_hist,0)
                    plt.plot(mm[:,0],label="err0")
                    plt.plot(mm[:,1],label="err1")
                    
                    nn=np.concatenate(ctrl_hist,0)
                    for c_i in range(3):
                        plt.plot(nn[:,c_i],label=f"ctrl_{c_i}")
                    plt.legend(fontsize=15);plt.show()

                    pdb.set_trace()
            
            data.ctrl[:] = - ctrl_c

            mujoco.mj_step(model, data)
            step_counter += 1
            viewer.sync()
