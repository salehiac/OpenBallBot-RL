import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np

import gymnasium as gym

import ballbotgym

def make_ballbot_env(gui=False,render_to_logs=False,test_only=False):
    def _init():
        env=gym.make(
                "ballbot-v0.1",
                GUI=gui,#should be disabled in parallel training
                renderer=render_to_logs,#this renders to logs, but is currently not supported for parallel envs. TODO: make the logs have an instance dependent name so it works
                apply_random_force_at_init=False,
                test_only=test_only,
                disable_cameras=True)#we disable cameras here since 1) the pid doesn't use them and 2) it considerably speeds up the simulation
        return env
    return _init


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

