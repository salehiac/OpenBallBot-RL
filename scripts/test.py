import numpy as np
import sys
import json
import pdb
import torch
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

sys.path.append("..")
from utils import make_ballbot_env

def main(args):

    if args.algo=="ppo":

        env=make_ballbot_env(gui=True,render_to_logs=True,test_only=True)()

        model = PPO.load(args.path)

        obs, _ = env.reset()
        done = False
       
        G_tau=0
        gamma=0.99999
        count=0
        while not done:
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic policy for testing
            print(action)
            obs, reward, done, truncated, info = env.step(action)

            G_tau+=gamma**count*reward
            count+=1
            print(f'step={count}')


        print("G_tau==",G_tau) 
        env.close()

if __name__=="__main__":

    _parser = argparse.ArgumentParser(description="Test a policy.")
    _parser.add_argument("--algo", type=str,help="choices are ppo, ...")
    _parser.add_argument("--path", type=str,help="path to policy")

    _args = _parser.parse_args()
    main(_args)


