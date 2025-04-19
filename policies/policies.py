import torch 
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
from abc import ABC, abstractmethod

sys.path.append("..")
import utils

class Policy(ABC):
    @abstractmethod
    def act(self,*args):
        pass


class PID(Policy):
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

        self.return_in_pitch_roll_space=False


    def act(self, R_mat:torch.tensor):

        with torch.no_grad():

            gravity=torch.tensor([0,0,-1.0]).float().reshape(3,1)
            gravity_local=R_mat.T.mm(gravity).reshape(3)
            up_axis_local=torch.tensor([0,0,1]).float()

            #all in local coordinates
            #error_vec=up_axis_local-(-gravity_local) #for the body to move in a direction, the ball should go in the opposite one
            #error_vec_2d=error_vec[:-1]
            error_vec_2d=torch.zeros(2)
            
            #pitch=torch.arcsin(-R_mat[2,0])
            #roll=torch.atan2(R_mat[2,1],R_mat[2,2])

            roll=torch.atan2(R_mat[2,1],R_mat[2,2]);
            pitch=torch.atan2(-R_mat[2,0],torch.sqrt(R_mat[2,1]**2+R_mat[2,2]**2));

            setpoint_r=setpoint_p=0
            error_vec_2d[0]=setpoint_p-pitch
            error_vec_2d[1]=setpoint_r-roll


            self.integral+=error_vec_2d*self.dt
            derivative=(error_vec_2d-self.prev_err)/self.dt
       
            #Now that we're computing pitch and roll error, this error is not in global coordinates but in roll/pitch space!
            u=self.k_p*error_vec_2d + self.k_i * self.integral + self.k_d * derivative


            self.prev_err=error_vec_2d

            angle_in_degrees=torch.acos(up_axis_local.dot(-gravity_local)).item()*180/np.pi


            #self.err_hist.append(angle_in_degrees)
            self.err_hist.append(error_vec_2d.reshape(1,2).numpy())

            up_axis_global=R_mat.mm(up_axis_local.reshape(3,1))

            #print(f"roll={roll}, pitch=={pitch}, up_axis_global={R_mat.mm(up_axis_local.reshape(3,1)).reshape(3)}, command (pitch/roll coords)=={u}")

            if self.return_in_pitch_roll_space:
                return u, angle_in_degrees
            else:#return in motor space
                ctrl_c=torch.zeros(3)
                ctrl_c[0]=u[1]*np.cos(utils.deg2rad(0))  + u[0]*np.sin(utils.deg2rad(0))
                ctrl_c[1]=u[1]*np.cos(utils.deg2rad(120))+ u[0]*np.sin(utils.deg2rad(120))
                ctrl_c[2]=u[1]*np.cos(utils.deg2rad(240))+ u[0]*np.sin(utils.deg2rad(240))
                ctrl_c=torch.clamp(ctrl_c,min=-10,max=10)

                return ctrl_c, angle_in_degrees

