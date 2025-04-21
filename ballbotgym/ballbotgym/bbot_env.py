import gymnasium as gym
import numpy as np
import argparse
import pdb
from typing import List
import sys
import time
import matplotlib.pyplot as plt
import quaternion
import os
import cv2
import string
from termcolor import colored

from scipy.linalg import logm

import mujoco
import mujoco.viewer


class RGBDInputs:

    def __init__(self,mjc_model, cam_name, height, width, normalize=True):

        self.width=width
        self.height=height
        
        self.renderer_rgb=mujoco.Renderer(mjc_model, width=width, height=height)
        self.renderer_d=mujoco.Renderer(mjc_model, width=width, height=height)
        self.renderer_d.enable_depth_rendering()

        self.cam_name=cam_name
        self.normalize=normalize

    def reset(self,mjc_model):

        self.renderer_rgb.close()
        self.renderer_d.close()
        
        self.renderer_rgb=mujoco.Renderer(mjc_model, width=self.width, height=self.height)
        self.renderer_d=mujoco.Renderer(mjc_model, width=self.width, height=self.height)
        self.renderer_d.enable_depth_rendering()

    def __call__(self, data):

        self.renderer_rgb.update_scene(data, camera=self.cam_name)  
        self.renderer_d.update_scene(data, camera=self.cam_name)  
        rgb=self.renderer_rgb.render().astype("float64")
        depth=np.expand_dims(self.renderer_d.render(),axis=-1)

        if self.normalize:
            rgb/=255.0
        return np.concatenate([rgb, depth],-1)

    def close(self):

        self.renderer_rgb.close()
        self.renderer_rgb=None
        self.renderer_d.close()
        self.renderer_d=None


class SceneRenderer:

    def __init__(self,model):

        self.framerate = 60  # (Hz)

        self.frames = []
        self.width=480
        self.height=480
        self.renderer=mujoco.Renderer(model,width=self.width, height=self.height)

    def reset(self, model, episode):

        self.renderer.close()
        self.renderer=mujoco.Renderer(model,width=self.width, height=self.height)
        self.frames=[]
        self.episode=episode

    def __call__(self, data):

        if len(self.frames) < data.time * self.framerate:
            self.renderer.update_scene(data)
            pixels = self.renderer.render()
            self.frames.append(pixels)

    def dump(self, path):

        #pdb.set_trace()
        dir_name=f"/episode_{self.episode}"
        if not os.path.exists(path+dir_name):
            os.mkdir(path+dir_name)
        counter=0
        for frame in self.frames:
            cv2.imwrite(path+dir_name+f"/frame_{counter}.png",   cv2.merge(cv2.split(frame)[::-1]))
            counter+=1


class BBotSimulation(gym.Env):

    def __init__(self,
            xml_path,
            GUI=False,#full mujoco gui
            renderer=True,#just scene render at 60fps
            apply_random_force_at_init=True,
            max_ep_steps=10000,
            im_shape={"h":128,"w":128},
            disable_cameras=False):
        """
        """
        super().__init__()

        self.xml_path= xml_path
        self.apply_random_force_at_init=apply_random_force_at_init
        self.max_ep_steps=max_ep_steps
        
        self.model=mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.rgbd_inputs=[RGBDInputs(self.model, cam_name="cam_0", height=im_shape["h"], width=im_shape["w"]), RGBDInputs(self.model, cam_name="cam_1", height=im_shape["h"], width=im_shape["w"])]
        self.passive_viewer=mujoco.viewer.launch_passive(self.model, self.data) if GUI else None
        self.scene_renderer=SceneRenderer(self.model) if renderer else None

        self.action_space=gym.spaces.Box(-1.0,1.0,shape=(3,),dtype=np.float64)
        self.observation_space=gym.spaces.Dict({
            "orientation": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float64),
            "angular_vel": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float64),
            "pos": gym.spaces.Box(low=-float("inf"),high=float("inf"), shape=(3,), dtype=np.float64),
            "vel": gym.spaces.Box(low=-float("inf"),high=float("inf"), shape=(3,), dtype=np.float64),
            "rgbd_0": gym.spaces.Box(low=0.0, high=1.0, shape=(im_shape["h"],im_shape["w"], 4), dtype=np.float64),
            "rgbd_1": gym.spaces.Box(low=0.0, high=1.0, shape=(im_shape["h"],im_shape["w"], 4), dtype=np.float64),
            }) if not disable_cameras else gym.spaces.Dict({
                "orientation": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float64),
                "angular_vel": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float64),
                "pos": gym.spaces.Box(low=-float("inf"),high=float("inf"), shape=(3,), dtype=np.float64),
                "vel": gym.spaces.Box(low=-float("inf"),high=float("inf"), shape=(3,), dtype=np.float64),
                })
        self.disable_cameras=disable_cameras

        self.goal_2d=self.model.geom("goal").pos[:-1]#goal to reach in the 2d plane. It is fixed in world coordinates, because we can learn a policy that pivots
                                                     #the robot in the desired direction so that the relative goal remains the same. 

        self.num_resets=0

        rand_str=''.join(np.random.permutation(list(string.ascii_letters + string.digits))[:12])
        self.log_dir="/tmp/log_"+rand_str
        os.mkdir(self.log_dir)

    @property
    def opt_timestep(self):
        return self.model.opt.timestep


    def reset(self,seed=None,goal:str="random",**kwargs):

        print("resetting_env...")

        super().reset(seed=seed)

        self.step_counter=0
        self.prev_data_time=0
        
       
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data) #recompute derivatives etc


        random_force = np.random.uniform(-10, 10, size=3) if self.apply_random_force_at_init else None
        #random_force=np.array([-11.77391146 , -3.03359904 ,  4.40871597])
        print("random_force applied at init (N)==",random_force)
        if random_force is not None:
            self.data.xfrc_applied[self.model.body("base").id, :3] = random_force

        self.prev_pos=None
        self.prev_orientation=None
        
        obs=self._get_obs()
        info=self._get_info()

        for rgbd_input in self.rgbd_inputs:
            rgbd_input.reset(self.model)
        if self.scene_renderer is not None:
            self.scene_renderer.reset(self.model,self.num_resets)

        self.G_tau=0.0
       
        self.num_resets+=1
        return obs, info

    def _get_obs(self):

        if not self.disable_cameras:
            rgbd_0=self.rgbd_inputs[0](self.data).astype("float64")
            rgbd_1=self.rgbd_inputs[1](self.data).astype("float64")

        #these would come from IMU or SLAM etc on a real system
        body_id = self.model.body("base").id  
        position = self.data.xpos[body_id].copy().astype("float64")
        orientation = quaternion.quaternion(*self.data.xquat[body_id].copy())
        rot_vec=quaternion.as_rotation_vector(orientation).astype("float64")

        #compute velocity
        vel=(position-self.prev_pos)/self.opt_timestep if self.prev_pos is not None else np.zeros_like(position)
        self.prev_pos=position
        #angular_vel=

        #compute angular velocity
        if self.prev_orientation is not None:
            #we compute the relative rotation between R_1 (orientation at previous timestamp) and R_2 (orientation at current timestamp)
            #this should give us the rotation that has happened between the two timestamps. All that is left is to map those to Rodriguez representations
            #and divide by the timestep
            R_1=quaternion.as_rotation_matrix(self.prev_orientation)
            R_2=quaternion.as_rotation_matrix(orientation)
            W=logm(R_1.T @ R_2).real
            vee = lambda S: np.array([S[2,1], S[0,2], S[1,0]])
            angular_vel=vee(W)/self.opt_timestep
        else:
            angular_vel=np.zeros_like(rot_vec)
        self.prev_orientation=orientation

        if self.disable_cameras:
            obs={"orientation":rot_vec, "angular_vel": angular_vel, "pos":position, "vel":vel}
        else:
            obs={"orientation":rot_vec, "angular_vel": angular_vel, "pos":position, "vel":vel, "rgbd_0":rgbd_0, "rgbd_1": rgbd_1}
        #print("obs==\n",obs)
        return obs
    
    def _get_info(self):
        
        info={"success":False,"failure":False,"step_counter":self.step_counter}

        return info
   
    def step(self, omniwheel_commands):
        """
        omniwheel_commands  np.tensor of shape(3,)
        """

        #print("commands==",omniwheel_commands)

        if self.data.time==0.0 and self.step_counter>0.0:#reset done in GUI
            print("RESET DUE TO RESET FROM GUI")
            self.reset()

        ctrl=omniwheel_commands*10
        ctrl=np.clip(ctrl,a_min=-10,a_max=10)#in case of pid issues
        
        self.data.ctrl[:] = - ctrl
        mujoco.mj_step(self.model, self.data)
        
        self.data.xfrc_applied[self.model.body("base").id, :3] = np.zeros(3)#this is to reset the initial force that is applied. From the documentation,
                                                                            #the force will be applied unless it is reset

        if self.scene_renderer is not None:
            self.scene_renderer(self.data)

        obs=self._get_obs()
        info=self._get_info()
        terminated=False
        truncated=False

        dist_to_goal=np.linalg.norm(self.goal_2d-obs["pos"][:-1])
        reward=-dist_to_goal#early fail penalty is added later
        #print("reward==",reward)-->

     
        #pdb.set_trace()

        if self.passive_viewer:
            self.passive_viewer.sync()

        self.prev_data_time=self.data.time#to detect resets that happen in GUI
        self.step_counter+=1

        if self.step_counter>=self.max_ep_steps:
            print(f"terminated. Cause: {self.step_counter}>self.max_ep_steps")
            terminated=True

        #compute angle with upright vector
        gravity=np.array([0,0,-1.0]).astype("float").reshape(3,1)
        gravity_local=(quaternion.as_rotation_matrix(quaternion.from_rotation_vector(obs["orientation"])).T @ (gravity)).reshape(3)

        up_axis_local=np.array([0,0,1]).astype("float")
        angle_in_degrees=np.arccos(up_axis_local.dot(-gravity_local)).item()*180/np.pi
       
        max_allowed_tilt=20
        if angle_in_degrees>max_allowed_tilt or obs["pos"][-1]<0.1:#if the robot is too tilted or if it has fallen (e.g. out of the bounds of the plane)
            print(f"failure after {self.step_counter}. Reason: tilte_angle > {max_allowed_tilt} ")

            info["success"]=False
            info["failure"]=True
            terminated=True
            early_fail_penalty=reward*(self.max_ep_steps-self.step_counter)
            reward+=early_fail_penalty


        elif dist_to_goal<0.01:

            info["success"]=True
            info["failure"]=False
            terminated=True

        #print("step_counter==",self.step_counter)
        if terminated and self.scene_renderer is not None:
            self.scene_renderer.dump(self.log_dir)

        gamma=1.0#not used algorithmically here anyway
        self.G_tau+=(gamma**self.step_counter)*reward
        if terminated:
            print(colored(f"G_tau=={self.G_tau}, num_steps=={self.step_counter}, reward=={reward-early_fail_penalty}, early_fail_penalty=={early_fail_penalty}","magenta",attrs=["bold"]))

        reward/=50000#normalization constant to avoid exploding losses
        return obs, reward, terminated, truncated, info


    def close(self):

        for i in range(len(self.rgbd_inputs)):
            self.rgbd_inputs[i].close()

        if self.passive_viewer:
            self.passive_viewer.close()

        del self.model
        del self.data



def main(args):

    sim=BBotSimulation(xml_path=args.xml_path,
            GUI=args.gui,
            max_ep_steps=args.max_steps,
            apply_random_force_at_init=True)

    obs, _=sim.reset()
    for step_i in range(sim.max_ep_steps):

        obs, reward, terminated, _, info=sim.step(sim.action_space.sample())





if __name__=="__main__":

    _parser = argparse.ArgumentParser(description="bbotgym test")
    _parser.add_argument("--xml_path", type=str)
    _parser.add_argument("--max_steps", type=int, default=5000)
    _parser.add_argument("--gui", action="store_true")

    _args = _parser.parse_args()
    main(_args)
