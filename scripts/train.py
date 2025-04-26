import numpy as np
import sys
import json
import pdb
import torch
import random
import argparse

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.noise import VectorizedActionNoise, NormalActionNoise

sys.path.append("..")
from utils import make_ballbot_env


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

def lr_schedule(progress_remaining):

    # progress_remaining goes from 1 (beginning) to 0 (end)
    return 5e-5 * progress_remaining



def main(args):


    N_ENVS = args.num_envs
    #vec_env = VecNormalize(SubprocVecEnv([make_ballbot_env() for _ in range(N_ENVS)]),norm_reward=True)
    vec_env = SubprocVecEnv([make_ballbot_env() for _ in range(N_ENVS)])
     
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[64, 64]))


    if args.algo=="ppo":
       
               
        #device is set to cpu because from the documentation, stabe_baseline_3's PPO is meant to run on cpu
        if not args.resume:
            model = PPO("MultiInputPolicy", 
                    vec_env,
                    verbose=1,
                    ent_coef=0.02,
                    device="cpu",
                    clip_range=0.1,#default is 0.2
                    vf_coef=0.5,#default i 0.5
                    learning_rate=5e-5,#lr_schedule,
                    policy_kwargs=policy_kwargs,
                    n_steps=2000)#n_steps means n_steps per env before update
        else:
            model=PPO.load(args.resume,device="cpu",env=vec_env)


        #pdb.set_trace()
        
        total_timesteps=2e6


    elif args.algo=="sac":

        policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

        normal_noise=NormalActionNoise(np.zeros(3),np.ones(3))
        vec_noise=VectorizedActionNoise(normal_noise,N_ENVS)
        if not args.resume:
            model = SAC("MultiInputPolicy", 
                    vec_env,
                    verbose=1,
                    learning_rate=1e-4,
                    #ent_coef=0.1,#let's keep the auto one which is proportional to reward
                    action_noise=vec_noise,
                    policy_kwargs=policy_kwargs,
                    device="cuda")
        else:
            model=SAC.load(args.resume,device="cuda",env=vec_env)

        total_timesteps=10e6
    
    else:
        raise Exception("Unknown algo")


    callback = CallbackList([
        ReturnLoggingCallback(N_ENVS),
        CheckpointCallback(
            save_freq=1000,               
            save_path=f"{args.out}/checkpoints/", 
            name_prefix="ppo_agent"     
            )
        ])

    model.learn(total_timesteps=total_timesteps,callback=callback)
        
    model.save(f"{args.out}/final_{args.algo}_agent")
    vec_env.close()



 

if __name__=="__main__":


       
    _parser = argparse.ArgumentParser(description="Train a policy.")
    _parser.add_argument("--algo", type=str,help="choices are ppo, ...",required=True)#algorithm params are hardcoded for ballbot
    _parser.add_argument("--num_envs", type=int, default=16)
    _parser.add_argument("--out", type=str, default="./log/", help="output path")
    _parser.add_argument("--resume", type=str, help="path to model",default="")

    _args = _parser.parse_args()

    repeatable=True
    if repeatable:
        _seed = 127
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)

        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(_seed)
    
    main(_args)

   
