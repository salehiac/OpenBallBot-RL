import numpy as np
import sys
import pdb
import matplotlib.pyplot as plt
import quaternion
import os

import torch
from termcolor import colored

sys.path.append("..")
import policies

import gymnasium as gym
import ballbotgym



if __name__=="__main__":
    if len(sys.argv)==3:
        _setpoint_p=float(sys.argv[1])*np.pi/180
        _setpoint_r=float(sys.argv[2])*np.pi/180
    else:
        _setpoint_r=_setpoint_p=0
    
    env=gym.make(
            "ballbot-v0.1",
            GUI=True,#full mujoco GUI
            goal_type="fixed_dir",#unusued for pid
            log_options={"cams":True,"reward_terms":False},
            disable_cameras=False)#we disable cameras here since 1) the pid doesn't use them and 2) it considerably speeds up the simulation
    
    #k_vals=[20,5,2] #works for 1khz
    k_vals=[20,15,2] #better for 500hz, but not optimal
    pid=policies.PID(dt=env.env.env.opt_timestep,
            k_p=k_vals[0],
            k_i=k_vals[1],
            k_d=k_vals[2])
    
    obs, _=env.reset()
    
    G_tau=0
    gamma=0.999999
    for step_i in range(env.env.env.max_ep_steps):
        
        ctrl,_=pid.act(
                torch.tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector(obs["orientation"][-3:]))).float(),
                setpoint_r=_setpoint_r,
                setpoint_p=_setpoint_p,
                )
        obs, reward, terminated, _, info=env.step(ctrl.numpy())
        G_tau+=gamma**step_i*reward
    
        #print(step_i,obs["orientation"])
        #if step_i>10:
        #    print(env.env.env.opt_timestep)
        #    break
    
        if terminated and not info["failure"]:#don't check for success here since success is defined w.r.t goal flag
            print(colored(f"successfuly balanced robot for {step_i} steps","green",attrs=["bold"]))
        elif terminated and info["failure"]:
            print(colored("failed!","red",attrs=["bold"]))
            break
    
    print("G_tau==",G_tau)
    env.env.env.close()
    import threading

    #print(threading.enumerate())
    os._exit(0)#not cool, but the passive viewer sometimes doesn't close properly
        
               
