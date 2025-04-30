import gymnasium as gym
import numpy as np
import argparse
import pdb
from typing import List
import sys
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import quaternion
import os
import cv2
import string
import re
from functools import reduce

from termcolor import colored
from scipy.linalg import logm

import mujoco
import mujoco.viewer

from . import Rewards

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
            goal_type:str,
            GUI=False,#full mujoco gui
            renderer=True,#just scene render at 60fps
            apply_random_force_at_init=True,
            im_shape={"h":128,"w":128},
            test_only=False,
            disable_cameras=False):
        """
        goal_type can be 'fixed_pos', 'fixed_dir', 'rand_pos', 'rand_dir', 'stop'
        test_only just gathers some additional debug data - TODO: remove it, I guess?
        """
        super().__init__()

        self.xml_path= xml_path
        self.goal_type=goal_type
        self.max_ep_steps=2500
        self.apply_random_force_at_init=apply_random_force_at_init

        self.action_space=gym.spaces.Box(-1.0,1.0,shape=(3,),dtype=np.float64)
        if goal_type=="fixed_pos":#uses position

            self.observation_space=gym.spaces.Dict({
                "orientation": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float64),
                "angular_vel": gym.spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float64),
                "pos": gym.spaces.Box(low=-float("inf"),high=float("inf"), shape=(3,), dtype=np.float64),
                "vel": gym.spaces.Box(low=-2,high=2, shape=(3,), dtype=np.float64),
                "rgbd_0": gym.spaces.Box(low=0.0, high=1.0, shape=(im_shape["h"],im_shape["w"], 4), dtype=np.float64),
                "rgbd_1": gym.spaces.Box(low=0.0, high=1.0, shape=(im_shape["h"],im_shape["w"], 4), dtype=np.float64),
                }) if not disable_cameras else gym.spaces.Dict({
                    "orientation": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float64),
                    "angular_vel": gym.spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float64),
                    "pos": gym.spaces.Box(low=-float("inf"),high=float("inf"), shape=(3,), dtype=np.float64),
                    "vel": gym.spaces.Box(low=-2,high=2, shape=(3,), dtype=np.float64),
                    })

        elif goal_type=="rand_dir":#no position, goal conditioned instead

            self.observation_space=gym.spaces.Dict({
                "orientation": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float64),
                "angular_vel": gym.spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float64),
                "goal": gym.spaces.Box(low=-1.0,high=1.0, shape=(2,), dtype=np.float64),
                "vel": gym.spaces.Box(low=-2,high=2, shape=(3,), dtype=np.float64),
                "rgbd_0": gym.spaces.Box(low=0.0, high=1.0, shape=(im_shape["h"],im_shape["w"], 4), dtype=np.float64),
                "rgbd_1": gym.spaces.Box(low=0.0, high=1.0, shape=(im_shape["h"],im_shape["w"], 4), dtype=np.float64),
                }) if not disable_cameras else gym.spaces.Dict({
                    "orientation": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float64),
                    "angular_vel": gym.spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float64),
                    "goal": gym.spaces.Box(low=-1.0,high=1.0, shape=(2,), dtype=np.float64),
                    "vel": gym.spaces.Box(low=-2,high=2, shape=(3,), dtype=np.float64),
                    })

        elif goal_type=="stop":
            
            raise Exception("not implemented yet")
        else: 
            raise Exception("unknown goal type {goal_type}")

        self.model=mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.rgbd_inputs=None
        if not disable_cameras:
            self.rgbd_inputs=[RGBDInputs(self.model, cam_name="cam_0", height=im_shape["h"], width=im_shape["w"]), RGBDInputs(self.model, cam_name="cam_1", height=im_shape["h"], width=im_shape["w"])]
        self.disable_cameras=disable_cameras

        self.passive_viewer=mujoco.viewer.launch_passive(self.model, self.data) if GUI else None
        self.scene_renderer=SceneRenderer(self.model) if renderer else None

        self.num_resets=0

        rand_str=''.join(np.random.permutation(list(string.ascii_letters + string.digits))[:12])
        self.log_dir="/tmp/log_"+rand_str
        os.mkdir(self.log_dir)

        self.test_only=test_only

        self.max_abs_obs={x:-1 for x in ["orientation", "angular_vel", "vel", "pos"]}

    @property
    def opt_timestep(self):
        return self.model.opt.timestep

    def _reset_goal_and_reward_objs(self):
        
        if self.goal_type=="rand_dir":
                
            self.goal_2d=sample_direction_uniform(num=1).reshape(2)
            self.reward_obj=Rewards.DirectionalReward(target_direction=self.goal_2d)

        elif self.goal_type=="fixed_pos":

            #we can learn a policy that pivots the robot in the desired direction so that the relative goal remains the same, hence the possibility of a fixed reward 
            self.goal_2d=[0.0, 0.5]
            self.reward_obj=Rewards.FixedReward(self.goal_2d[1])
            #self.reward_obj.plot_reward()

        else:
            raise Exception("unknown goal type")


        if self.test_only:
            self.reward_hist=[]

        print("goal_2d==",self.goal_2d)

    def reset(self,seed=None,goal:str="random",**kwargs):

        print("resetting_env...")

        super().reset(seed=seed)

        self._reset_goal_and_reward_objs()

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

        if not self.disable_cameras:
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
        vel=np.clip(vel,a_min=-2.0,a_max=2.0)
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
            angular_vel=np.clip(vee(W)/self.opt_timestep,a_min=-2.0,a_max=2.0)
        else:
            angular_vel=np.zeros_like(rot_vec)
        self.prev_orientation=orientation

        if self.goal_type=="fixed_pos":
            if self.disable_cameras:
                obs={"orientation":rot_vec, "angular_vel": angular_vel, "pos":position, "vel":vel}
            else:
                obs={"orientation":rot_vec, "angular_vel": angular_vel, "pos":position, "vel":vel, "rgbd_0":rgbd_0, "rgbd_1": rgbd_1}

        elif self.goal_type=="rand_dir":

            if self.disable_cameras:
                obs={"orientation":rot_vec, "angular_vel": angular_vel, "goal":self.goal_2d, "vel":vel}
            else:
                raise Exception("this is not handled yet")
     
        #print("obs==",obs)
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
        #print("ctrl==",ctrl)
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

        reward=self.reward_obj(obs)#note that failure penalties are added later
        reward= reward/1000 if self.goal_type=="rand_dir" else reward/100 #normalization to get better gradients
     
        if self.passive_viewer:

             
            with self.passive_viewer.lock():

                if self.goal_type=="rand_dir":
                    self.passive_viewer.user_scn.ngeom=1#yeah, =1, not +=1
                    factor=20#just for display
                    mujoco.mjv_initGeom(
                            self.passive_viewer.user_scn.geoms[self.passive_viewer.user_scn.ngeom-1],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.1]*3,
                            pos=[self.goal_2d[0]*factor,self.goal_2d[1]*factor,0.0],
                            mat=np.eye(3).flatten(),
                            rgba=[1, 0, 1, 1])

                    mujoco.mjv_connector(
                            self.passive_viewer.user_scn.geoms[self.passive_viewer.user_scn.ngeom-1],
                            type=mujoco.mjtGeom.mjGEOM_LINE,
                            width=200,
                            from_=[0,0,0],
                            to=[self.goal_2d[0]*factor,self.goal_2d[1]*factor,0])
                elif self.goal_type=="fixed_pos":
                    self.passive_viewer.user_scn.ngeom=1#yeah, =1, not +=1
                    mujoco.mjv_initGeom(
                            self.passive_viewer.user_scn.geoms[self.passive_viewer.user_scn.ngeom-1],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.01]*3,
                            pos=[self.goal_2d[0],self.goal_2d[1],0.0],
                            mat=np.eye(3).flatten(),
                            rgba=[1, 0, 1, 1])

                    mujoco.mjv_connector(
                            self.passive_viewer.user_scn.geoms[self.passive_viewer.user_scn.ngeom-1],
                            type=mujoco.mjtGeom.mjGEOM_LINE,
                            width=10,
                            from_=[self.goal_2d[0],self.goal_2d[1],0],
                            to=[self.goal_2d[0],self.goal_2d[1],0.8])
 


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
        early_fail_penalty=0.0
        if angle_in_degrees>max_allowed_tilt:
            print(f"failure after {self.step_counter}. Reason: tilte_angle > {max_allowed_tilt} ")

            info["success"]=False
            info["failure"]=True
            terminated=True
            early_fail_penalty=-1.0
            reward+=early_fail_penalty

        #print("step_counter==",self.step_counter)
        if terminated and self.scene_renderer is not None:
            self.scene_renderer.dump(self.log_dir)

        gamma=1.0#not used algorithmically here anyway
        self.G_tau+=(gamma**self.step_counter)*reward
        if terminated:
            print(colored(f"G_tau=={self.G_tau}, num_steps=={self.step_counter}, reward=={reward-early_fail_penalty}, early_fail_penalty=={early_fail_penalty}","magenta",attrs=["bold"]))

            if self.test_only:
                #remove this?
                pass
                #import matplotlib.pyplot as plt
                #plt.plot(self.reward_hist,"b")
                #plt.show()


        if self.test_only:
            self.reward_hist.append(reward)


        #print("reward==",reward)
        return obs, reward, terminated, truncated, info


    def close(self):

        if not self.disable_cameras:
            for i in range(len(self.rgbd_inputs)):
                self.rgbd_inputs[i].close()

        if self.passive_viewer:
            self.passive_viewer.close()

        #this just freezes for some reason
        #if glfw.get_current_context() is not None:
        #    glfw.terminate()

        del self.model
        del self.data

def sample_direction_uniform(num=1):

    t=np.random.rand(num).reshape(num,1)*2*np.pi
    return np.concatenate([np.cos(t),np.sin(t)],1)





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
    _parser.add_argument("--gui", action="store_true")

    _args = _parser.parse_args()
    main(_args)
