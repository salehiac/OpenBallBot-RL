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
        GUI=True,
        max_ep_steps=10000,
        apply_random_force_at_init=True,
        disable_cameras=True)#we disable cameras here since 1) the pid doesn't use them and 2) it considerably speeds up the simulation

k_vals=[200,50,20]
pid=policies.PID(dt=env.env.env.opt_timestep,
        k_p=k_vals[0],
        k_i=k_vals[1],
        k_d=k_vals[2])

obs, _=env.reset()
for step_i in range(env.env.env.max_ep_steps):
    
    ctrl,_=pid.act(torch.tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector(obs["orientation"]))).float())
    obs, reward, terminated, _, info=env.step(ctrl.numpy())

print(colored(f"successfuly balanced robot for {step_i} steps","green",attrs=["bold"]))
env.env.env.close()
    
           
