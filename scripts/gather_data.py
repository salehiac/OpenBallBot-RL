import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np
import argparse
import pdb
import sys

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO, SAC
import gymnasium as gym
from termcolor import colored

import ballbotgym
sys.path.append("..")
import policies



def make_env():
    env=gym.make(
                "ballbot-v0.1",
                GUI=False,#should be disabled in parallel training
                goal_type="fixed_dir",
                depth_only=True,
                test_only=False,
                disable_cameras=False,
                log_options={"cams":True,"reward_terms":False})
    return env

def main(args):
  
    num_envs=args.n_envs
    vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
    print(colored(f"num_envs={num_envs}\nData will be written to {vec_env.get_attr('log_dir')}","yellow",attrs=["bold"]))
    obs = vec_env.reset()
    
    model = PPO.load(args.policy)

    #pdb.set_trace()
    for s_i in range(args.n_steps):
           
        action = model.predict(obs, deterministic=True)  
        obs, _ , _, _ = vec_env.step(action[0])

    vec_env.close()
    
 


if __name__=="__main__":
    
    _parser = argparse.ArgumentParser(description="Gather depth images for encoder pretraining")
    _parser.add_argument("--n_steps", type=int,help="number of timesteps per environment to gather")
    _parser.add_argument("--n_envs", type=int,help="number of timesteps per environment to gather")
    _parser.add_argument("--policy", type=str, help="the path to a compatible PPO policy.")

    _args = _parser.parse_args()
    main(_args)


