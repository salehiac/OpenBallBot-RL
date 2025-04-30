import numpy as np
from abc import ABC, abstractmethod
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import quaternion



class Reward(ABC):


    @abstractmethod
    def __call__(self, state):
        pass


class DirectionalReward(Reward):

    def __init__(self, target_direction):
        """
        target_direction   np.array of shape (2,)
        """

        self.target_direction=target_direction

    def __call__(self, state):



        dir_rew=state["vel"][:-1].dot(self.target_direction)

        R_mat=quaternion.as_rotation_matrix(quaternion.from_rotation_vector(state["orientation"]))#local to global

        #adding an "alive" bonus at each timestep seems to work better than having an upright bonus
        #robot_up_axis_local=np.array([0,0,1]).astype("float").reshape(3,1)
        #robot_up_axis_global_coords=R_mat.T@robot_up_axis_local
        #upright_rew=(np.array([0,0,1]).reshape(3,1).T@robot_up_axis_global_coords).item()

        #coef=0.7#should be in [0,1]
        #rew=(1-coef)*dir_rew+coef*upright_rew

        rew=dir_rew
        
        #print(f"dir_rew=={dir_rew},    upright_rew={upright_rew},    total_rew={rew}")
        return rew

class FixedReward(Reward):

    def __init__(self, goal_y):
        """
        assumes a goal that is on the x axis
        """
        super().__init__()
        self.goal_y=goal_y

    def __call__(self,state):


        pos2d=state["pos"][:-1]

        dx=0.1
        dy=0.1

        if (pos2d[0]<-dx or pos2d[0]>dx) or (pos2d[1]<0 or pos2d[1]>self.goal_y):
            return 0.0

        reward=np.clip(pos2d[1]/self.goal_y,a_min=0.0,a_max=1.0)

        return reward

    def plot_reward(self,min_x=-2.0,max_x=2.0,min_y=-2.0,max_y=2.0):
        """
        debug function
        """

        x = np.linspace(min_x,max_x, 500)
        y = np.linspace(min_y,max_y, 500)
        x, y = np.meshgrid(x, y)
        z=np.zeros_like(x)
        for ii in range(x.shape[0]):
            for jj in range(x.shape[1]):
                z[ii,jj]=self({"pos":[x[ii,jj],y[ii,jj],0]})
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()
 
