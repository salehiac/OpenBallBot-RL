import numpy as np
import sys
import pdb
import matplotlib.pyplot as plt
import quaternion

import torch
from termcolor import colored

sys.path.append("..")
import policies

import gymnasium as gym
import ballbotgym

env=gym.make(
        "ballbot-v0.1",
        GUI=True,#full mujoco GUI
        renderer=True,#renders to log
        max_ep_steps=10000,
        apply_random_force_at_init=True,
        disable_cameras=True)#we disable cameras here since 1) the pid doesn't use them and 2) it considerably speeds up the simulation

k_vals=[20,5,2]
pid=policies.PID(dt=env.env.env.opt_timestep,
        k_p=k_vals[0],
        k_i=k_vals[1],
        k_d=k_vals[2])

obs, _=env.reset()

G_tau=0
gamma=0.999999
for step_i in range(env.env.env.max_ep_steps):
    
    ctrl,_=pid.act(torch.tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector(obs["orientation"]))).float())
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
    
           
