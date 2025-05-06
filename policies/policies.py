import torch 
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
from abc import ABC, abstractmethod
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

sys.path.append("..")
import utils

class Policy(ABC):
    @abstractmethod
    def act(self,*args):
        pass



class Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        
        for key, subspace in observation_space.spaces.items():
            
            if "rgbd_" in key:
              
                #note that we're iterating on observation_space objects, so there is not batch size info
                C,H,W=subspace.shape #typically, C=1 and H=W=32 here
                F1=8
                F2=8
                out_sz=16
                extractors[key] = torch.nn.Sequential(
                    torch.nn.Conv2d(1, F1, kernel_size=3, stride=2, padding=1), #output BxF1xH/2xW/2
                    torch.nn.BatchNorm2d(F1),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(F1, F2, kernel_size=3, stride=2, padding=1), #output BxF2xH/4xW/4
                    torch.nn.BatchNorm2d(F2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Flatten(),                                       
                    torch.nn.Linear(F2*H//4*W//4, out_sz),                                  
                    torch.nn.BatchNorm1d(out_sz),
                    torch.nn.LeakyReLU(),
                    )

                total_concat_size += 16
            else:
                #note that we're iterating on observation_space objects, so there is not batch size info
                S=subspace.shape[0]
                extractors[key] = torch.nn.Flatten()
                total_concat_size += S

        self.extractors = torch.nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        encoded_tensor_dict={}#for debug only

        for key, extractor in self.extractors.items():
            cur=extractor(observations[key])
            
            #if "rgbd_" in key:
            #    print(observations[key])
            #encoded_tensor_dict[key]=cur
            
            encoded_tensor_list.append(cur)
      
        #for k,v in encoded_tensor_dict.items():
        #    print(k,v)
        return torch.cat(encoded_tensor_list, dim=1)

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


    def act(self, R_mat:torch.tensor,setpoint_r=0, setpoint_p=0):
        """
        setpoint_p  pitch target
        setpoint_r  roll target
        """


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

