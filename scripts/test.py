import numpy as np
import sys
import json
import pdb
import torch
import argparse
import random
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from termcolor import colored

sys.path.append("..")
from utils import make_ballbot_env

def main(args,seed=None):


    with torch.no_grad():
        if args.algo=="ppo":
            model = PPO.load(args.path)
        elif args.algo=="sac":
            model=SAC.load(args.path)
        else:
            raise Exception("unknown algo")
        
        env=make_ballbot_env(gui=True,
                terrain_type=model.terrain_type if not args.override_terrain_type else args.override_terrain_type,
                log_options={"cams":False, "reward_terms":True},
                seed=seed)()
        print(f"bbot mass is {sum(env.env.env.env.model.body_mass)}")
        if args.override_terrain_type:
            print(colored(f"policy was trained on terrain type {model.terrain_type}, but it will be tested on terrain type {args.override_terrain_type}","yellow",attrs=["bold"]))
        else:
            print(colored(f"policy was trained on terrain type {model.terrain_type} and will be tested on the same terrain type.", "yellow",attrs=["bold"]))


        p_sum=sum([param.abs().sum().item() for param in model.policy.parameters() if param.requires_grad])
        print(colored(f"sum_of_model_params=={p_sum}","yellow"))

        for test_i in range(args.n_test):
            obs, _ = env.reset(seed=seed+test_i)
            done = False
               
            G_tau=0
            gamma=0.99999
            count=0
            while not done:
                action, _ = model.predict(obs, deterministic=True)  # Use deterministic policy for testing
                #print(action)
                obs, reward, done, truncated, info = env.step(action)

                G_tau+=gamma**count*reward
                count+=1
                #print(f'step={count}')


            print("G_tau==",G_tau) 

    env.close()
    
    return model

if __name__=="__main__":

    _parser = argparse.ArgumentParser(description="Test a policy.")
    _parser.add_argument("--algo", type=str,help="choices are ppo, ...")
    _parser.add_argument("--path", type=str,help="path to policy")
    _parser.add_argument("--n_test", type=int,help="How many times to test policy",default=1)
    _parser.add_argument("--seed", type=int,help="For repeatablility. If not set, will be chosen randomly",default=-1)
    _parser.add_argument("--override_terrain_type", type=str,default="", help="Can be used to run the policy on a terrain type it hasn't been trained on. See the ballbot env for options.")

    _args = _parser.parse_args()
 
    _seed=_args.seed if _args.seed!=-1 else np.random.randint(10000)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)

    _model=main(_args,seed=_seed)

