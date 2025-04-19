import gymnasium as gym
import numpy as np
import argparse
import pdb
from typing import List
import sys
import time
import matplotlib.pyplot as plt
import quaternion

import mujoco
import mujoco.viewer


class RGBDInputs:

    def __init__(self,mjc_model, cam_name, height, width, normalize=True):

        self.renderer_rgb=mujoco.Renderer(mjc_model, width=width, height=height)
        self.renderer_d=mujoco.Renderer(mjc_model, width=width, height=height)
        self.renderer_d.enable_depth_rendering()

        self.cam_name=cam_name
        self.normalize=normalize

    def __call__(self, data):

        self.renderer_rgb.update_scene(data, camera=self.cam_name)  
        self.renderer_d.update_scene(data, camera=self.cam_name)  
        rgb=self.renderer_rgb.render().astype("float64")
        depth=np.expand_dims(self.renderer_d.render(),axis=-1)

        if self.normalize:
            rgb/=255.0
        return np.concatenate([rgb, depth],-1)

    def close(self):

        self.renderer_rgb=None
        self.renderer_d=None


class BBotSimulation(gym.Env):

    def __init__(self,
            xml_path,
            GUI=False,
            apply_random_force_at_init=True,
            max_ep_steps=5000,
            im_shape={"h":480,"w":480},
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

        self.action_space=gym.spaces.Box(-float("inf"),float("inf"),shape=(3,),dtype=np.float64)
        self.observation_space=gym.spaces.Dict({
            "R_mat": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,3), dtype=np.float64),
            "pos": gym.spaces.Box(low=-float("inf"),high=float("inf"), shape=(3,), dtype=np.float64),
            "rgbd_0": gym.spaces.Box(low=0.0, high=1.0, shape=(im_shape["h"],im_shape["w"], 4), dtype=np.float64),
            "rgbd_1": gym.spaces.Box(low=0.0, high=1.0, shape=(im_shape["h"],im_shape["w"], 4), dtype=np.float64),
            }) if not disable_cameras else gym.spaces.Dict({
                "R_mat": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,3), dtype=np.float64),
                "pos": gym.spaces.Box(low=-float("inf"),high=float("inf"), shape=(3,), dtype=np.float64),
                })
        self.disable_cameras=disable_cameras


    @property
    def opt_timestep(self):
        return self.model.opt.timestep


    def reset(self,seed=None,goal:str="random",**kwargs):

        super().reset(seed=seed)

        self.step_counter=0
        self.prev_data_time=0
        
       
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data) #recompute derivatives etc


        random_force = np.random.uniform(-15, 15, size=3) if self.apply_random_force_at_init else None
        print("random_force applied at init (N)==",random_force)
        self.data.xfrc_applied[self.model.body("base").id, :3] = random_force

        obs=self._get_obs()
        info=self._get_info()
        
        
        return obs, info

    def _get_obs(self):

        if not self.disable_cameras:
            rgbd_0=self.rgbd_inputs[0](self.data).astype("float64")
            rgbd_1=self.rgbd_inputs[1](self.data).astype("float64")

        #these would come from IMU or SLAM etc on a real system
        body_id = self.model.body("base").id  
        position = self.data.xpos[body_id].astype("float64")
        orientation = quaternion.quaternion(*self.data.xquat[body_id])
  
        R_mat=quaternion.as_rotation_matrix(orientation).astype("float64")

        if self.disable_cameras:
            obs={"R_mat":R_mat, "pos":position}
        else:
            obs={"R_mat":R_mat, "pos":position, "rgbd_0":rgbd_0, "rgbd_1": rgbd_1}
        return obs
    
    def _get_info(self):
        
        info={"success":False,"failure":False}

        return info
   
    def step(self, omniwheel_commands):
        """
        omniwheel_commands  np.tensor of shape(3,)
        """

        if self.data.time==0.0 and self.step_counter>0.0:#reset done in GUI
            print("RESET DUE TO RESET FROM GUI")
            self.reset()

        ctrl=np.clip(omniwheel_commands,a_min=-10,a_max=10)
        
        self.data.ctrl[:] = - ctrl
        mujoco.mj_step(self.model, self.data)

        obs=self._get_obs()
        info=self._get_info()
        reward=0.0
        terminated=False
        truncated=False

        if self.passive_viewer:
            self.passive_viewer.sync()

        self.prev_data_time=self.data.time#to detect resets that happen in GUI
        self.step_counter+=1

        if self.step_counter>self.max_ep_steps:
            print(f"terminated. Cause: {self.step_counter}>self.max_ep_steps")
            terminated=True

        #compute angle with upright vector
        gravity=np.array([0,0,-1.0]).astype("float").reshape(3,1)
        gravity_local=(obs["R_mat"].T @ (gravity)).reshape(3)

        up_axis_local=np.array([0,0,1]).astype("float")
        angle_in_degrees=np.arccos(up_axis_local.dot(-gravity_local)).item()*180/np.pi
       
        max_allowed_tilt=20
        if angle_in_degrees>max_allowed_tilt:
            print(f"failure after {self.step_counter}. Reason: tilte_angle > {max_allowed_tilt} ")

            info["success"]=False
            info["failure"]=True
            terminated=True

        #print("step_counter==",self.step_counter)

        return obs, reward, terminated, truncated, info


    def plot_obs(self):

        pass

        #plt.ion()
        #cam_fig_ax[0,0].imshow(rgbd_0[:,:,:3])
        #cam_fig_ax[0,1].imshow(rgbd_0[:,:,3])
        #cam_fig_ax[1,0].imshow(rgbd_1[:,:,:3])
        #cam_fig_ax[1,1].imshow(rgbd_1[:,:,3])
        #plt.pause(0.0001)
        #plt.show()


    def close(self):

        for i in range(len(self.rgbd_inputs)):
            self.rgbd_inputs[i].close()

        if self.passive_viewer:
            self.passive_viewer.close()



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
