import numpy as np
import sys
import json
import pdb
import torch
import random
import os
import argparse
import shutil

import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import VectorizedActionNoise, NormalActionNoise
from stable_baselines3.common.logger import configure
from termcolor import colored

sys.path.append("..")
from utils import make_ballbot_env
import policies


def lr_schedule(progress_remaining):

    # progress_remaining goes from 1 (beginning) to 0 (end)

    if progress_remaining>0.7:
        return 1e-4
    elif progress_remaining <0.7 and progress_remaining>0.5:
        return 5e-5
    else:
        return 1e-5

def confirm(question):
    inpt=""
    while inpt!='y' and inpt!='n':
        inpt=input(question+" [y/N]: ").strip().lower()
    return inpt=='y'


def main(config,seed):


   
    if config["algo"]["name"]=="ppo":

        policy_kwargs = dict(
            activation_fn=torch.nn.LeakyReLU,
            net_arch=dict(
                pi=[config["hidden_sz"], config["hidden_sz"], config["hidden_sz"], config["hidden_sz"]],
                vf=[config["hidden_sz"], config["hidden_sz"], config["hidden_sz"], config["hidden_sz"]]),
            features_extractor_class=policies.Extractor,#note that this will be shared by the policy and the value networks
            features_extractor_kwargs={"frozen_encoder_path":config["frozen_cnn"]},
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={"weight_decay":float(config["algo"]["weight_decay"])},
            )


       
        N_ENVS = int(config["num_envs"])

        vec_env  = SubprocVecEnv([make_ballbot_env(goal_type=config["goal_type"],terrain_type=config["problem"]["terrain_type"],seed=seed) for _ in range(N_ENVS)])
        eval_env = SubprocVecEnv([make_ballbot_env(goal_type=config["goal_type"],terrain_type=config["problem"]["terrain_type"],seed=seed+1) for _ in range(N_ENVS)])



        #even though stabe_baseline_3's PPO is primarily meant to run on cpu (per their documentation), the CNN runs like 10x times faster on GPU, so...
        device="cuda"
        if not config["resume"]:

            normalize_advantage=bool(config["algo"]["normalize_advantage"])
            model = PPO("MultiInputPolicy", 
                    vec_env,
                    verbose=1,
                    ent_coef=float(config["algo"]["ent_coef"]),
                    device=device,
                    clip_range=float(config["algo"]["clip_range"]),
                    target_kl=float(config["algo"]["target_kl"]),
                    vf_coef=float(config["algo"]["vf_coef"]),
                    learning_rate=float(config["algo"]["learning_rate"]) if config["algo"]["learning_rate"]!=-1 else lr_schedule,
                    policy_kwargs=policy_kwargs,
                    n_steps=int(config["algo"]["n_steps"]),
                    batch_size=int(config["algo"]["batch_sz"]),
                    n_epochs=int(config["algo"]["n_epochs"]),
                    normalize_advantage=normalize_advantage,
                    seed=seed)

            #pdb.set_trace()
        else:
            print(colored(f"loading model from {config['resume']}...", "yellow", attrs=["bold"]))

            custom_objects=dict(
                    ent_coef=float(config["algo"]["ent_coef"]),
                    device=device,
                    clip_range=float(config["algo"]["clip_range"]),
                    vf_coef=float(config["algo"]["vf_coef"]),
                    learning_rate=float(config["algo"]["learning_rate"]) if config["algo"]["learning_rate"]!=-1 else lr_schedule,
                    n_steps=int(config["algo"]["n_steps"]),
                    seed=seed)

            for k,v in custom_objects.items():
                print(k,v)

            #pdb.set_trace()
            model=PPO.load(config["resume"],device=device,env=vec_env,custom_objects=custom_objects)

            #for param_group in model.policy.optimizer.param_groups:
            #    param_group['lr'] = float(config["algo"]["learning_rate"])


        #pdb.set_trace()
        
        total_timesteps=int(float(config["total_timesteps"]))

        
       
    elif config["algo"]["name"]=="sac":

        N_ENVS = int(config["num_envs"])
        vec_env = SubprocVecEnv([make_ballbot_env(goal_type=config["goal_type"],terrain_type=config["problem"]["terrain_type"],seed=seed) for _ in range(N_ENVS)])
        eval_env = SubprocVecEnv([make_ballbot_env(goal_type=config["goal_type"],terrain_type=config["problem"]["terrain_type"],seed=seed+1) for _ in range(N_ENVS)])


        policy_kwargs = dict(
            activation_fn=torch.nn.LeakyReLU,
            net_arch=dict(
                pi=[config["hidden_sz"], config["hidden_sz"], config["hidden_sz"], config["hidden_sz"]],
                qf=[config["hidden_sz"], config["hidden_sz"], config["hidden_sz"], config["hidden_sz"]]),
            features_extractor_class=policies.Extractor,#note that this will be shared by the policy and the value networks
            features_extractor_kwargs={"frozen_encoder_path":config["frozen_cnn"]},
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={"weight_decay":float(config["algo"]["weight_decay"])},
            )



        normal_noise=NormalActionNoise(np.zeros(3),np.ones(3)*float(config["algo"]["action_noise_sigma"]))
        vec_noise=VectorizedActionNoise(normal_noise,N_ENVS)
        if not config["resume"]:
            model = SAC("MultiInputPolicy", 
                    vec_env,
                    verbose=1,
                    learning_rate=float(config["algo"]["learning_rate"]),
                    ent_coef="auto_0.1",
                    #action_noise=vec_noise,
                    policy_kwargs=policy_kwargs,
                    device="cuda",
                    batch_size=config["algo"]["batch_size"],
                    buffer_size=config["algo"]["buffer_size"],
                    seed=seed)
        else:
            model=SAC.load(config["resume"],device="cuda",env=vec_env)

        total_timesteps=int(float(config["total_timesteps"]))
    
    else:
        raise Exception("Unknown algo")

    if os.path.exists(config['out']):
        if confirm(colored(f"The output directory ({config['out']}) specified in your yaml file already exists. Overwrite?","red",attrs=["bold"])):
            shutil.rmtree(config["out"])  
        else:
            print(colored("Okay, aborted! Exiting.","red"))
            os._exit(1)
    
    os.mkdir(config['out'])




    with open(f"{config['out']}/config.yaml","w") as fl:
        yaml.dump(config,fl)
    with open(f"{config['out']}/info.txt","w") as fl:
        #for retrocompatibility
        json.dump({"algo": config["algo"]["name"], "num_envs": config["num_envs"] , "out": config["out"], "resume": config["resume"], "seed": config["seed"]},fl)


    logger_path= f"{config['out']}/"
    logger= configure(logger_path, ["stdout", "csv"])
    model.set_logger(logger)

    eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f'{config["out"]}/best_model',
            log_path=f"{config['out']}/results/",
            eval_freq=5000 if config["algo"]["name"]=="ppo" else 500,
            n_eval_episodes=8,
            deterministic=True,
            render=False,
            )

    callback = CallbackList([
        eval_callback,
        CheckpointCallback(
            20000,
            save_path=f"{config['out']}/checkpoints/", 
            name_prefix=f"{config['algo']['name']}_agent"     
            )
        ])

    if config["algo"]["name"]=="ppo":
        print(model.policy)
        num_params_total=sum([param.numel() for param in model.policy.parameters()])
        num_params_learnable=sum([param.numel() for param in model.policy.parameters() if param.requires_grad])
        print(colored(f"num_total_params={num_params_total}","cyan",attrs=["bold"]))
        print(colored(f"num_learnable_params={num_params_learnable}","cyan",attrs=["bold"]))
        print(model.policy.optimizer)
        print(colored(f"total_timesteps={total_timesteps}","yellow"))
        num_updates_per_rollout=(config["algo"]["n_epochs"]*config["num_envs"]*config["algo"]["n_steps"])/config["algo"]["batch_sz"]
        if not confirm(colored(f"the current config results in {num_updates_per_rollout} number of updates per rollout. Continue? ","green",attrs=["bold"])):
            print("Aborting.")
            os._exit(1)
        else:
            print("Okay.")
    #pdb.set_trace()    

    model.terrain_type=config["problem"]["terrain_type"]
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
    
        main(_config,seed=_seed)
    else:
        raise Exception("nah you want this to be repeatable")

   
