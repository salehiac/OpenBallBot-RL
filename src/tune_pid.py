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
from utils import plot_vectors
from assets import OmniWheelRef


    
def run_simulation(model,pid):
    data = mujoco.MjData(model)
    omni=OmniWheelRef()


    step_c=-1
    for _ in range(4000):
        
        mujoco.mj_step(model, data)

        body_id = model.body("base").id  
        position = data.xpos[body_id]  
        orientation = quaternion.quaternion(*data.xquat[body_id])

        #print(step_c)
        #print("position==",position,"orientation==", orientation)
        #pdb.set_trace()

        with torch.no_grad():
            R_mat=quaternion.as_rotation_matrix(orientation)
            ctrl_b,angle_err=pid.act(torch.tensor(R_mat).float())
            if angle_err>15.0:
                #err_hist=np.concatenate(pid.err_hist,1)#2*N
                #int_hist=np.concatenate(pid.integral_hist,1)
                #der_hist=np.concatenate(pid.derivative_hist,1)
                #plt.plot(err_hist[0,:],label="err_0")
                #plt.plot(err_hist[1,:],label="err_1")

                #plt.plot(int_hist[0,:],label="int_0")
                #plt.plot(int_hist[1,:],label="int_1")

                #plt.plot(der_hist[0,:],label="der_0")
                #plt.plot(der_hist[1,:],label="der_1")
                #plt.legend(fontsize=16)
                #plt.show()
                #pdb.set_trace()
                return False, step_c
                

        ctrl_c=omni.body_plane_to_control_space(ctrl_b.cpu().detach().numpy().reshape(2,1)).reshape(3)
        data.ctrl[:] = ctrl_c

       
        step_c+=1

    return True, step_c

if __name__=="__main__":

    _model = mujoco.MjModel.from_xml_path(sys.argv[1])

    #reversed friction
    #kp=[100]
    #ki=np.linspace(60,100,100)
    #kd=np.linspace(60,100,100)
    
    kp=[-500]
    ki=np.linspace(-50,0,500)
    kd=np.linspace(-50,0,500)

    successes=[]

    num_trials=len(kp)*len(ki)*len(kd)
    trial_i=0
    best_len=-float("inf")
    best_config=[]
    for k_p in kp:
        for k_i in ki:
            for k_d in kd:
                if trial_i%100==0 and trial_i:
                    print(f"trial {trial_i}/{num_trials}")
                    print(best_len,best_config)

                pid=policies.PID(dt=_model.opt.timestep,k_p=k_p,k_i=k_i,k_d=k_d)
                result,ep_len=run_simulation(model=_model,pid=pid)
                if result:
                    print(colored("*********************************** SUCCESS! ************************** ","green",attrs=["bold"]))
                    successes.append([k_p,k_i,k_d])
                trial_i+=1

                if ep_len>best_len:
                    best_len=ep_len
                    best_config=[k_p,k_i,k_d]



