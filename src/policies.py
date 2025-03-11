import torch 
import numpy as np
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

    def __init__(self):
        super().__init__()

        self.k_p=torch.tensor([
            [100,100,100],
            [100,100,100],
            [100,100,100]]).float()
        self.k_d=1
        self.k_i=1

        self.num_act=0


    def act(self, R_mat:torch.tensor):
        """
        R_mat rotation matrix
        """
        gravity=torch.tensor([0,0,-9.8]).float()
        up_axis_local=torch.tensor([[0,0,1]]).reshape(3,1).float()
        up_axis_global=R_mat.mm(up_axis_local).reshape(3)

        print("axis_up_glb==",up_axis_global)

        error=torch.linalg.cross(up_axis_global,-gravity)
        print("error_vec==",error)

        u=self.k_p.mm(error.reshape(3,1))

        self.num_act+=1
        if self.num_act==2050:
            pdb.set_trace()

        return u.reshape(3), error.norm()







