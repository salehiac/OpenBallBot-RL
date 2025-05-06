import numpy as np
import sys
import json
import pdb
import torch
import random
import argparse

import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.noise import VectorizedActionNoise, NormalActionNoise
from stable_baselines3.common.logger import configure
from termcolor import colored

sys.path.append("..")
from utils import make_ballbot_env
import policies



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

        self.log_dir=log_dir
        self.best_mean_rew=-float("inf")

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

        with open(self.log_dir+"/G_tau_lst","w") as fl:
            json.dump({"avg_returns":self.returns_report,"total_steps":self.steps_report},fl)

        return True

    def _on_rollout_end(self):

        rb=self.locals["rollout_buffer"]
        mean_reward=rb.rewards.mean()#bb.rewards is of shame n_steps*num_envs for ppo

        if mean_reward > self.best_mean_rew:
                self.best_mean_rew= mean_reward
                self.model.save(self.log_dir+"/best_agent")


def lr_schedule(progress_remaining):

    # progress_remaining goes from 1 (beginning) to 0 (end)
    return 5e-5 * progress_remaining


def main(config):


   
    if config["algo"]["name"]=="ppo":


        policy_kwargs = dict(
            activation_fn=torch.nn.LeakyReLU,
            net_arch=dict(
                pi=[config["hidden_sz"], config["hidden_sz"], config["hidden_sz"]],
                vf=[config["hidden_sz"], config["hidden_sz"], config["hidden_sz"]]),
            features_extractor_class=policies.Extractor,#note that this will be shared by the policy and the value networks
            features_extractor_kwargs={"frozen_encoder_path":config["frozen_cnn"]},
            )


        N_ENVS = int(config["num_envs"])

        vec_env = SubprocVecEnv([make_ballbot_env(goal_type=config["goal_type"]) for _ in range(N_ENVS)])

        #even though stabe_baseline_3's PPO is primarily meant to run on cpu (per their documentation), the CNN runs like 10x times faster on GPU, so...
        device="cuda"
        if not config["resume"]:
            model = PPO("MultiInputPolicy", 
                    vec_env,
                    verbose=1,
                    ent_coef=float(config["algo"]["ent_coef"]),
                    device=device,
                    clip_range=float(config["algo"]["clip_range"]),
                    vf_coef=float(config["algo"]["vf_coef"]),
                    learning_rate=float(config["algo"]["learning_rate"]) if config["algo"]["learning_rate"]!=-1 else lr_schedule,
                    policy_kwargs=policy_kwargs,
                    n_steps=int(config["algo"]["n_steps"]))
        else:
            model=PPO.load(config["resume"],device=device,env=vec_env)

        #pdb.set_trace()
        
        total_timesteps=5e6

        
       
    elif config["algo"]["name"]=="sac":

        N_ENVS = int(config["num_envs"])
        vec_env = SubprocVecEnv([make_ballbot_env(goal_type=config["goal_type"]) for _ in range(N_ENVS)])

        #policy_kwargs = dict(net_arch=dict(
        #    activation_fn=torch.nn.LeakyReLU,
        #    pi=[config["hidden_sz"], config["hidden_sz"]],
        #    qf=[config["hidden_sz"], config["hidden_sz"]]))

        normal_noise=NormalActionNoise(np.zeros(3),np.ones(3)*float(config["algo"]["action_noise_sigma"]))
        vec_noise=VectorizedActionNoise(normal_noise,N_ENVS)
        if not config["resume"]:
            model = SAC("MultiInputPolicy", 
                    vec_env,
                    verbose=1,
                    learning_rate=float(config["algo"]["learning_rate"]),
                    ent_coef="auto_0.1",
                    #action_noise=vec_noise,
                    #policy_kwargs=policy_kwargs,
                    device="cuda")
        else:
            model=SAC.load(config["resume"],device="cuda",env=vec_env)

        total_timesteps=5e6
    
    else:
        raise Exception("Unknown algo")


    with open(f"{config['out']}/config.yaml","w") as fl:
        json.dump(config,fl)
    with open(f"{config['out']}/info.txt","w") as fl:
        #for retrocompatibility
        json.dump({"algo": config["algo"]["name"], "num_envs": config["num_envs"] , "out": config["out"], "resume": config["resume"], "seed": config["seed"]},fl)


    logger_path= f"{config['out']}/"
    logger= configure(logger_path, ["stdout", "csv"])
    model.set_logger(logger)

    callback = CallbackList([
        ReturnLoggingCallback(N_ENVS,log_dir=f"{config['out']}/"),
        CheckpointCallback(
            save_freq=1000 if config["algo"]["name"]!="ppo" else 10000,
            save_path=f"{config['out']}/checkpoints/", 
            name_prefix=f"{config['algo']['name']}_agent"     
            )
        ])

    print(model.policy)
    num_params_total=sum([param.numel() for param in model.policy.parameters()])
    num_params_learnable=sum([param.numel() for param in model.policy.parameters() if param.requires_grad])
    print(colored(f"num_total_params={num_params_total}","cyan",attrs=["bold"]))
    print(colored(f"num_learnable_params={num_params_learnable}","cyan",attrs=["bold"]))
    #pdb.set_trace()    
    model.learn(total_timesteps=total_timesteps,callback=callback)
        
    vec_env.close()

if __name__=="__main__":


    _parser = argparse.ArgumentParser(description="Train a policy.")
    _parser.add_argument("--config", help="your yaml config file", required=True)

    _args = _parser.parse_args()

    with open(_args.config, "r") as f:
        _config = yaml.safe_load(f)

    repeatable=True if int(_config["seed"])!=-1 else False
    if repeatable:
        _seed = int(_config["seed"])
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
    
    main(_config)

   
