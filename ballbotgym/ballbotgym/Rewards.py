import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import quaternion

class DirectionalReward:

    def __init__(self, target_direction):
        """
        target_direction   np.array of shape (2,)
        """

        self.target_direction=target_direction

    def __call__(self, state):

        dir_rew=state["vel"][-3:-1].dot(self.target_direction)
        rew=dir_rew

        return rew


 
