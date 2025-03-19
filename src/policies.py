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

        gravity=torch.tensor([0,0,-1.0]).float().reshape(3,1)
        gravity_local=R_mat.T.mm(gravity).reshape(3)
        up_axis_local=torch.tensor([0,0,1]).float()

        #all in local coordinates
        error_vec=up_axis_local-(-gravity_local) #for the body to move in a direction, the ball should go in the opposite one
        error_vec_2d=error_vec[:-1]

        self.integral+=error_vec_2d*self.dt
        derivative=(error_vec_2d-self.prev_err)/self.dt
        u=self.k_p*error_vec_2d + self.k_i * self.integral + self.k_d * derivative


        self.prev_err=error_vec_2d

        angle_in_degrees=torch.acos(up_axis_local.dot(-gravity_local)).item()*180/np.pi

        self.err_hist.append(angle_in_degrees)

        #debug stuff
        u_full=torch.zeros(3,1)
        u_full[:2,0]=u
        u_full_global=R_mat.mm(u_full)

        up_axis_global=R_mat.mm(up_axis_local.reshape(3,1))


        return u, angle_in_degrees , u_full_global, up_axis_global#in local coordinates #in local coordinates



    def act_garbage(self, R_mat:torch.tensor):
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







