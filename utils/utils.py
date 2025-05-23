import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np

from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

import ballbotgym

def make_ballbot_env(
        terrain_type,
        gui=False,
        disable_cams=False,
        seed=0,
        log_options={"cams":False, "reward_terms":False},
        eval_env=False):
    """
    eval_env is just to ensure repeatability with stable_baselines3. During training, stable_baselines3 seeds the env at init (with base_seed+i where i is in [0,num_envs]), but it doesn't do
    so during eval (seed is always None, and so no _np_random is created). To ensure that we can reproduce the terrains used during eval, we pass a seed to the env which will then manually create
    _np_random.
    
    Therefore, it's not required to be set eval_env=True if you're testing a policy from outside stablebaseline3's framework.
    """
    def _init():
        env=gym.make(
                "ballbot-v0.1",
                GUI=gui,#should be disabled in parallel training
                log_options=log_options,
                terrain_type=terrain_type,
                eval_env=[eval_env,seed])#because stablebaselines's EvalCallback, in contrast with training, doesn't seed at the first iteration

        return Monitor(env) #using a Monitor wrapper to enable logging rollout avg rewards 
    return _init

def deg2rad(d):

    return d*np.pi/180

def rad2deg(r):

    return r*180/np.pi

