import gymnasium as gym
import numpy as np
import sys
import json
import pdb

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback


import bbot2d

class ReturnLoggingCallback(BaseCallback):
    def __init__(self, num_envs,verbose=0):
        super().__init__(verbose)
        self.num_envs=num_envs
        self.G_tau_lst=[0.0]*self.num_envs
        self.gamma=0.9999#it's for logging, not very important 

        self.full_episode_returns=[]
        self.returns_report=[]

    def _on_step(self):

        for e_i in range(self.num_envs):
            rew=self.locals["rewards"][e_i]
            env_steps=self.locals["infos"][e_i]["num_steps"]
            self.G_tau_lst[e_i]+=(self.gamma**env_steps)*rew
            if self.locals["dones"][e_i]:
                self.full_episode_returns.append(self.G_tau_lst[e_i])
                self.G_tau_lst[e_i]=0.0

        while len(self.full_episode_returns)>=self.num_envs:
            avg_ret=np.mean(self.full_episode_returns[:self.num_envs])
            self.full_episode_returns=self.full_episode_returns[self.num_envs:]
            self.returns_report.append(avg_ret)

        with open("/tmp/G_tau_report","w") as fl:
            json.dump(self.returns_report,fl)

        return True


def make_env(render=False):
    def _init():
        return gym.make(
                "bbot2d-v0.1",
                no_ball=False,
                render=render,
                continuous_actions=False,
                max_ep_steps=400)

    return _init
        

if __name__=="__main__":

    if sys.argv[1]=="train":
       
        N_ENVS = 16  # Adjust based on available compute
        vec_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
        
        # Define PPO model
        model = PPO("MultiInputPolicy", vec_env, verbose=1,ent_coef=0.01)
        
        # Train the model
        callback = ReturnLoggingCallback(N_ENVS)
        model.learn(total_timesteps=800000,callback=callback)
        #model.learn(total_timesteps=200000)
        
        # Save and test the model
        model.save("ppo_bbot")
        vec_env.close()

    elif sys.argv[1]=="test":

        env=make_env(render=True)()

        model = PPO.load(sys.argv[2])

        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic policy for testing
            obs, reward, done, truncated, info = env.step(action)
        
        env.close()
