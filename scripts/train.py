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
from stable_baselines3.common.logger import configure

sys.path.append("..")
from utils import make_ballbot_env


class ReturnLoggingCallback(BaseCallback):
    def __init__(self, num_envs, log_dir, verbose=0):
        super().__init__(verbose)
        self.num_envs=num_envs
        self.G_tau_lst=[0.0]*self.num_envs
        self.gamma=0.9999#it's for logging, not very important 

        self.full_episode_returns=[]
        self.returns_report=[]

        self.num_steps_total=0
        self.steps_report=[]

        self.log_path=log_dir+"/G_tau_lst"

    def _on_step(self):

        for e_i in range(self.num_envs):
            rew=self.locals["rewards"][e_i]
            env_steps=self.locals["infos"][e_i]["step_counter"]
            self.G_tau_lst[e_i]+=(self.gamma**env_steps)*rew
            if self.locals["dones"][e_i]:
                self.full_episode_returns.append(self.G_tau_lst[e_i])
                self.G_tau_lst[e_i]=0.0

            self.num_steps_total+=1

        while len(self.full_episode_returns)>=self.num_envs:
            avg_ret=np.mean(self.full_episode_returns[:self.num_envs])
            self.full_episode_returns=self.full_episode_returns[self.num_envs:]
            self.returns_report.append(avg_ret)
            self.steps_report.append(self.num_steps_total)

        with open(self.log_path,"w") as fl:
            json.dump({"avg_returns":self.returns_report,"total_steps":self.steps_report},fl)

        return True


def main(args):


        
    #policy_kwargs = dict(activation_fn=torch.nn.Tanh,
    policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU,
            net_arch=dict(pi=[512, 512], vf=[512, 512]))


    if args.algo=="ppo":

        N_ENVS = args.num_envs
        vec_env = SubprocVecEnv([make_ballbot_env(goal_type="fixed") for _ in range(N_ENVS)])

       
               
        #device is set to cpu because from the documentation, stabe_baseline_3's PPO is meant to run on cpu
        if not args.resume:
            model = PPO("MultiInputPolicy", 
                    vec_env,
                    verbose=1,
                    ent_coef=0.02,
                    device="cpu",
                    clip_range=0.1,#default is 0.2
                    vf_coef=0.5,#default i 0.5
                    learning_rate=5e-5,
                    policy_kwargs=policy_kwargs,
                    n_steps=2000)#n_steps means n_steps per env before update
        else:
            model=PPO.load(args.resume,device="cpu",env=vec_env)


        #pdb.set_trace()
        
        total_timesteps=5e6

        ppo_log_path= f"{args.out}/"
        ppo_logger= configure(ppo_log_path, ["stdout", "csv"])
        model.set_logger(ppo_logger)

        with open(f"{args.out}/info.txt","w") as fl:
            json.dump(args.__dict__,fl)


    elif args.algo=="sac":

        N_ENVS = args.num_envs
        vec_env = SubprocVecEnv([make_ballbot_env(goal_type="fixed") for _ in range(N_ENVS)])

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
        ReturnLoggingCallback(N_ENVS,log_dir=f"{args.out}/"),
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
    _parser.add_argument("--seed", type=int, help="For repeatability/debug. Passing -1 (which is default) disables this. Args used in paper are 0,127,45,31,871",default=-1)

    _args = _parser.parse_args()

    repeatable=True if _args.seed!=-1 else False
    if repeatable:
        _seed = _args.seed
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

   
