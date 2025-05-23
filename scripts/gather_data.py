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
from utils import make_ballbot_env

def main(args):
  
    num_envs=args.n_envs
    vec_env = SubprocVecEnv([make_ballbot_env(goal_type="fixed_dir",seed=args.seed,log_options={"cams":True,"reward_terms":False}) for _ in range(num_envs)])
    vec_env.seed(args.seed)
    obs = vec_env.reset()
    
    print(colored(f"num_envs={num_envs}\nData will be written to {vec_env.get_attr('log_dir')}","yellow",attrs=["bold"]))
    model = PPO.load(args.policy)

    for s_i in range(args.n_steps):
           
        action = model.predict(obs, deterministic=True)  
        obs, _ , _, _ = vec_env.step(action[0])

    vec_env.close()
    
 


if __name__=="__main__":
    
    _parser = argparse.ArgumentParser(description="Gather depth images for encoder pretraining")
    _parser.add_argument("--n_steps", type=int,help="number of timesteps per environment to gather")
    _parser.add_argument("--n_envs", type=int,help="number of timesteps per environment to gather")
    _parser.add_argument("--policy", type=str, help="the path to a compatible PPO policy.")
    _parser.add_argument("--seed", type=int, default=0)

    _args = _parser.parse_args()
    main(_args)


