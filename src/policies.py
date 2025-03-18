import torch 
import numpy as np
import matplotlib.pyplot as plt
import pdb
from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def act(self,*args):
        pass


class PID(ABC):
    """
    doesn't support batch simulations
    """

    def __init__(self,dt,k_p,k_i,k_d):
        super().__init__()

        self.k_p=k_p
        self.k_i=k_i
        self.k_d=k_d

        self.dt=dt

        self.integral=torch.zeros(2)
        self.prev_err=torch.zeros(2)

        self.err_hist=[]
        self.integral_hist=[]
        self.derivative_hist=[]



    def act(self, R_mat:torch.tensor):
        """
        R_mat rotation matrix
        """
        #gravity=torch.tensor([0,0,-9.8]).float()
        gravity=torch.tensor([0,0,-1.0]).float()
        up_axis_local=torch.tensor([[0,0,1]]).reshape(3,1).float()
        up_axis_global=R_mat.mm(up_axis_local).reshape(3)

        #print("axis_up_glb==",up_axis_global)

        angle_in_degrees=np.arcsin(np.linalg.norm(torch.linalg.cross(up_axis_global,-gravity)))*180/np.pi
        error=-up_axis_global[:-1]

       
        self.integral+=error*self.dt
        derivative=(error-self.prev_err)/self.dt
        self.prev_err=error

        u=self.k_p*error+self.k_i*self.integral+self.k_d*derivative
        
        self.err_hist.append(angle_in_degrees)
        self.integral_hist.append(self.integral)
        self.derivative_hist.append(derivative)

        #pdb.set_trace()

        return u, angle_in_degrees







