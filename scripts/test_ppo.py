import gymnasium as gym
import numpy as np
import sys
import json
import pdb
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback


import ballbotgym

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
            env_steps=self.locals["infos"][e_i]["step_counter"]
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


def make_env(gui=False,render_to_logs=False):
    def _init():
        env=gym.make(
                "ballbot-v0.1",
                GUI=gui,#should be disabled in parallel training
                renderer=render_to_logs,#this renders to logs, but is currently not supported for parallel envs. TODO: make the logs have an instance dependent name so it works
                max_ep_steps=20000,
                apply_random_force_at_init=False,
                disable_cameras=True)#we disable cameras here since 1) the pid doesn't use them and 2) it considerably speeds up the simulation
        return env
    return _init
        

if __name__=="__main__":

    if sys.argv[1]=="train":
       
        N_ENVS = 16 # Adjust based on available compute
        vec_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
        
        # Define PPO model
        #model = PPO("MultiInputPolicy", vec_env, verbose=1,ent_coef=0.01,device="cpu",learning_rate=1e-5)
        model = PPO("MultiInputPolicy", vec_env, verbose=1,use_sde=True,device="cpu",learning_rate=1e-5)
        #pdb.set_trace()
        
        # Train the model
        callback = CallbackList([
            ReturnLoggingCallback(N_ENVS),
            CheckpointCallback(
                save_freq=10000,               # Save every N timesteps
                save_path="./log/checkpoints/",    # Directory to save models
                name_prefix="ppo_agent"        # File name prefix
                )
            ])


        #pdb.set_trace()
        model.learn(total_timesteps=10000000,callback=callback)
        #model.learn(total_timesteps=200000)
        
        # Save and test the model
        model.save("./log/final_ppo_agent")
        vec_env.close()

    elif sys.argv[1]=="test":

        env=make_env(gui=True,render_to_logs=True)()

        model = PPO.load(sys.argv[2])

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


        print("G_tau==",G_tau) 
        env.close()
