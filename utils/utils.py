import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np

from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

import ballbotgym

def make_ballbot_env(
        goal_type,
        gui=False,
        test_only=False,
        disable_cams=False,
        seed=0):
    """
    goal_type can be 'fixed_pos', 'fixed_dir', 'rand_dir', 'rand_pos', 'stop'
    """
    def _init():
        env=gym.make(
                "ballbot-v0.1",
                GUI=gui,#should be disabled in parallel training
                goal_type=goal_type,
                test_only=test_only)
        if seed is not None:
            env_s=SeedWrapper(env,seed=seed)
        return Monitor(env) #using a Monitor wrapper to enable logging rollout avg rewards 
    return _init

class SeedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._seed = seed

    def reset(self,seed=None):
        if seed is None:
            return self.env.reset(seed=self._seed)
        return self.env.reset(seed=seed)


def deg2rad(d):

    return d*np.pi/180

def rad2deg(r):

    return r*180/np.pi


def rad_sec_to_rpm(rs):

    return 60*rs/(2*np.pi)

def deg_sec_to_rpm(ds):
    
    return 60*ds/360

def rpm_to_deg_sec(rpm):

    return rpm*360/60

def plot_vectors(origins: np.ndarray,
                 directions: np.ndarray,
                 colors: List[str],
                 fig_ax: Optional[List] = None,
                 axis_limits: List[float] = None,
                 scale_factor: float = 1.0,
                 clear: bool = True,
                 dashed: bool = False,
                 no_pause: bool = False):
    """
    origins  should be of shape num_pts*3, optionally, it can also be 1*3. In that case, all vectors will have the same origin
    directions  should be of shape num_pts*3
    """
    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]

    if clear:
        ax.clear()
    ax.quiver(*(origins.T * scale_factor),
              *(directions.T * scale_factor),
              color=colors,
              linestyle="solid" if not dashed else "dashed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    axis_limits = axis_limits if axis_limits is not None else [10.0] * 3
    ax.set_xlim([-axis_limits[0], axis_limits[0]])
    ax.set_ylim([-axis_limits[1], axis_limits[1]])
    ax.set_zlim([-axis_limits[2], axis_limits[2]])

    if not no_pause:
        plt.pause(0.01)
    return [fig, ax]

