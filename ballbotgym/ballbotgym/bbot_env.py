import gymnasium as gym
import numpy as np
import random
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
from dataclasses import dataclass
import shutil

from termcolor import colored
from scipy.linalg import logm

import mujoco
import mujoco.viewer

from . import Rewards, terrain


_default_dtype=np.float32

class RGBDInputs:


    def __init__(self,mjc_model, 
            height,
            width,
            cams:List[str],
            disable_rgb:bool):

        self.width=width
        self.height=height
       
        self._renderer_rgb=mujoco.Renderer(mjc_model, width=width, height=height) if not disable_rgb else None
        self._renderer_d=mujoco.Renderer(mjc_model, width=width, height=height)
        self._renderer_d.enable_depth_rendering()

        self.cams=cams

    def __call__(self, data, cam_name:str):

        assert cam_name in self.cams, f"wrong cam name (got {cam_name}, available ones are {self.cams})"

        if self._renderer_rgb is not None:
            self._renderer_rgb.update_scene(data, camera=cam_name)  
            rgb=self._renderer_rgb.render().astype(_default_dtype)/255
       
        self._renderer_d.update_scene(data, camera=cam_name)  
        
        depth=np.expand_dims(self._renderer_d.render(),axis=-1)

        depth[depth>=1.0]=1.0#with robot orientations - especially near failure - the depth cameras might catch a glimpse of the sky which will return large values 
                             #adjusting zfar in the xml is one solution, but since the actual clipping plane will be zfar*model.stats.extent, with the latter's computation not 
                             #super transparent, I prefer to do some additional clipping here 

        #plt.imshow(rgb);plt.title(cam_name);plt.show()
        #plt.imshow(depth);plt.title(cam_name);plt.show()
        #pdb.set_trace()

        if self._renderer_rgb is not None:
            arr=np.concatenate([rgb, depth],-1)
        else:
            arr=depth

        return arr

    def reset(self,mjc_model):

        self._renderer_d.close()
        if self._renderer_rgb is not None:
            self._renderer_rgb.close()
            self._renderer_rgb=mujoco.Renderer(mjc_model, width=self.width, height=self.height)

        self._renderer_d=mujoco.Renderer(mjc_model, width=self.width, height=self.height)
        self._renderer_d.enable_depth_rendering()

    def close(self):

        if self._renderer_rgb is not None:
            self._renderer_rgb.close()
            self._renderer_rgb=None
        self._renderer_d.close()
        self._renderer_d=None

@dataclass
class StampedImPair:
    im_0: np.ndarray
    im_1: np.ndarray
    ts: float


class BBotSimulation(gym.Env):

    def __init__(self,
            xml_path,
            GUI=False,#full mujoco gui
            im_shape={"h":64,"w":64},
            disable_cameras=False,
            depth_only=True,
            log_options={"cams":False,"reward_terms":False},
            max_ep_steps=None,
            terrain_type:str="perlin",
            eval_env=[False,None]):
        """
        terrain_type ca be "perlin", "flat"
        """
        super().__init__()

        self.terrain_type=terrain_type
        
        self.xml_path= xml_path 
        self.max_ep_steps=4000 if max_ep_steps is None else max_ep_steps
        self.camera_frame_rate=90#in Hz
        self.log_options=log_options
        self.depth_only=depth_only

        self.action_space=gym.spaces.Box(-1.0,1.0,shape=(3,),dtype=_default_dtype)

        num_channels=1 if self.depth_only else 4
        self.observation_space=gym.spaces.Dict({
            "orientation": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=_default_dtype),
            "angular_vel": gym.spaces.Box(low=-2, high=2, shape=(3,), dtype=_default_dtype),
            "vel": gym.spaces.Box(low=-2,high=2, shape=(3,), dtype=_default_dtype),
            "motor_state": gym.spaces.Box(-2.0,2.0,shape=(3,),dtype=_default_dtype),
            "actions": gym.spaces.Box(-1.0,1.0,shape=(3,),dtype=_default_dtype),
            "rgbd_0": gym.spaces.Box(low=0.0, high=1.0, shape=(num_channels, im_shape["h"], im_shape["w"]), dtype=_default_dtype),
            "rgbd_1": gym.spaces.Box(low=0.0, high=1.0, shape=(num_channels, im_shape["h"], im_shape["w"]), dtype=_default_dtype),
            "relative_image_timestamp": gym.spaces.Box(low=0.0, high=0.1, shape=(1,), dtype=_default_dtype),#the lag between high-frequencey proprio and low-freq cams
            }) if not disable_cameras else gym.spaces.Dict({
                "orientation": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=_default_dtype),
                "angular_vel": gym.spaces.Box(low=-2, high=2, shape=(3,), dtype=_default_dtype),
                "vel": gym.spaces.Box(low=-2,high=2, shape=(3,), dtype=_default_dtype),
                "motor_state": gym.spaces.Box(-2.0,2.0,shape=(3,),dtype=_default_dtype),
                "actions": gym.spaces.Box(-1.0,1.0,shape=(3,),dtype=_default_dtype),
                "relative_image_timestamp": gym.spaces.Box(low=0.0, high=0.1, shape=(1,), dtype=_default_dtype),#the lag between high-frequencey proprio and low-freq cams
                })

        self.model=mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.rgbd_inputs=None
        if not disable_cameras:
            self.rgbd_inputs=RGBDInputs(self.model,  height=im_shape["h"], width=im_shape["w"], cams=["cam_0", "cam_1"], disable_rgb=self.depth_only)
            self.prev_im_pair=StampedImPair(im_0=None,im_1=None, ts=0)
            if self.log_options["cams"]:
                self.rgbd_hist_0=[]
                self.rgbd_hist_1=[]
        self.disable_cameras=disable_cameras

        self.passive_viewer=mujoco.viewer.launch_passive(self.model, self.data) if GUI else None

       
        self.log_dir=None #set in reset

        self.max_abs_obs={x:-1 for x in ["orientation", "angular_vel", "vel", "pos"]}


        self.reward_term_1_hist=[]
        self.reward_term_2_hist=[]
        self.reward_term_3_hist=[]#constant for now

        self.num_episodes=-1
        self.eval_env=eval_env[0]

        if self.eval_env:
            self._np_random, _ = gym.utils.seeding.np_random(eval_env[1])

        self.verbose=False

    def effective_camera_frame_rate(self):

        #see comment on framerate in self._get_obs
        dt_mj=self.opt_timestep
        desired_cam_dt=1/self.camera_frame_rate

        N=np.ceil(desired_cam_dt/dt_mj)

        effective_framre_rate=1.0/(N*dt_mj)

        return effective_framre_rate


    @property
    def opt_timestep(self):
        return self.model.opt.timestep

    def _reset_goal_and_reward_objs(self):
        

        self.goal_2d=[0.0, 1.0]
        self.reward_obj=Rewards.DirectionalReward(target_direction=self.goal_2d)

    def _reset_terrain(self):

        nrows=self.model.hfield_nrow.item()
        ncols=self.model.hfield_ncol.item() 

        assert nrows==ncols, f"terrain is expected to have an equal number of rows an cols (got {nrows}, {ncols} in xml file)"
        assert self.model.hfield_size[0,0]==self.model.hfield_size[0,1], f"terrain should have equal length and width (got {self.model.hfield_size[0,:2]} in xml file)"

        sz=self.model.hfield_size[0,0]
        hfield_height_coef=self.model.hfield_size[0,2]

        if self.terrain_type=="perlin":
            r_seed=self._np_random.integers(0,10000)
            #print("r_seed==",r_seed)

            self.last_r_seed=r_seed
            self.model.hfield_data=terrain.generate_perlin_terrain(nrows,seed=r_seed)
        elif self.terrain_type=="flat":
            self.model.hfield_data=np.zeros(nrows**2)
        else:
            raise Exception("unknown terrain type")

        if self.passive_viewer is not None:
            self.passive_viewer.update_hfield(mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_HFIELD,"terrain"))

        #doing this once to get robot bounding boxes etc
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data) #recompute derivatives etc
      
         
        ball_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,"the_ball")
        assert self.data.geom_xpos[ball_id][0]==0 and self.data.geom_xpos[ball_id][1]==0, "terrain generation assumes that the ball is centered at (0,0)"
        ball_pos = self.data.geom_xpos[ball_id]
        ball_size = self.model.geom_size[ball_id]
        aabb_min = ball_pos - ball_size[0]#note that ball_size is specified with only one element in the xml
        aabb_max = ball_pos + ball_size[0]

        cell_size=sz/nrows
        center_idx=nrows//2

        terrain_mat=self.model.hfield_data.reshape(nrows,ncols)
        sub_terr=terrain_mat[
                center_idx-abs(int(aabb_min[0]//cell_size)): center_idx+int(aabb_max[0]//cell_size)+1,
                center_idx-abs(int(aabb_min[1]//cell_size)): center_idx+int(aabb_max[1]//cell_size)+1]

        eps=0.01
        init_robot_height_offset=sub_terr.max()*hfield_height_coef+eps

        #pdb.set_trace()

        return init_robot_height_offset

       
    def reset(self,seed=None,goal:str="random",**kwargs):
        
        #print(f"resetting_env, seed={seed}, eval_env={self.eval_env}")

        super().reset(seed=seed)

        self._reset_goal_and_reward_objs()

        self.step_counter=0
        self.prev_data_time=0

        init_robot_height_offset=self._reset_terrain()
      
        mujoco.mj_resetData(self.model, self.data)
        self.data.joint("base_free_joint").qpos[2]+=init_robot_height_offset
        self.data.joint("ball_free_joint").qpos[2]+=init_robot_height_offset
        mujoco.mj_forward(self.model, self.data) #recompute derivatives etc

        self.rgbd_inputs.reset(self.model)#otherwise it wont get updated with the new hfield
        self.prev_im_pair=StampedImPair(im_0=None,im_1=None, ts=0)

        self.prev_pos=None
        self.prev_orientation=None
        self.prev_motor_state=None
        self.prev_time=0


        obs=self._get_obs(np.zeros(3).astype(_default_dtype))
        info=self._get_info()

        if not self.disable_cameras and self.log_options["cams"]:
            self.rgbd_hist_0=[]
            self.rgbd_hist_1=[]

        self.G_tau=0.0
        self.num_episodes+=1

        if self.num_episodes==0 and self.verbose:
            print(colored(f"effective_frame_rate=={self.effective_camera_frame_rate()}","cyan",attrs=["bold"]))


        if self.log_dir is None and not self.eval_env:
            rand_str=''.join(self._np_random.permutation(list(string.ascii_letters + string.digits))[:12])
            self.log_dir="/tmp/log_"+rand_str
            if os.path.exists(self.log_dir):
                print(f"log_dir {self.log_dir} already exists. Overwriting!")
                shutil.rmtree(self.log_dir)  
            else:
                print(f"Creating log_dir {self.log_dir}")
            os.mkdir(self.log_dir)

        return obs, info

    def _save_logs(self):

        if self.eval_env:
            return


        if not self.disable_cameras and self.log_options["cams"]:
            if not self.disable_cameras:
                dir_name=f"{self.log_dir}/rgbd_log_episode_{self.num_episodes}"
                dir_name_rgb=dir_name+"/rgb/"
                dir_name_d=dir_name+"/depth/"
                #if not os.path.exists(dir_name):
                os.mkdir(dir_name)
                os.mkdir(dir_name_d)
                os.mkdir(dir_name_rgb)


                for ii in range(len(self.rgbd_hist_0)):
                    if not self.depth_only:
                        cv2.imwrite(f"{dir_name_rgb}/rbgd_a_{ii}.png",(cv2.merge(cv2.split(self.rgbd_hist_0[ii][:,:,:3])[::-1])*255).astype("uint8"))
                        cv2.imwrite(f"{dir_name_rgb}/rbgd_b_{ii}.png",(cv2.merge(cv2.split(self.rgbd_hist_1[ii][:,:,:3])[::-1])*255).astype("uint8"))
                        cv2.imwrite(f"{dir_name_d}/depth_a_{ii}.png",(self.rgbd_hist_0[ii][:,:,3]*255).astype("uint8"))
                        cv2.imwrite(f"{dir_name_d}/depth_b_{ii}.png",(self.rgbd_hist_1[ii][:,:,3]*255).astype("uint8"))
                    else:
                        cv2.imwrite(f"{dir_name_d}/depth_a_{ii}.png",(self.rgbd_hist_0[ii]*255).astype("uint8"))
                        cv2.imwrite(f"{dir_name_d}/depth_b_{ii}.png",(self.rgbd_hist_1[ii]*255).astype("uint8"))

        if self.log_options["reward_terms"]:
            if len(self.reward_term_1_hist):
                np.save(self.log_dir+"/term_1",np.array(self.reward_term_1_hist))
                np.save(self.log_dir+"/term_2",np.array(self.reward_term_2_hist))

        if self.terrain_type=="perlin":
            with open(f"{self.log_dir}/terrain_seed_history", "a") as fl:
                fl.write(f"{self.last_r_seed}\n")


    def _get_obs(self,last_ctrl):

        if not self.disable_cameras:

            delta_time=self.data.time-self.prev_im_pair.ts
            #delta_mujoco=self.data.time-self.prev_time #confirmed that mujoco is running at 500hz with current xml config 
            #self.prev_time=self.data.time
          
            #the following condition generally results in less frames than a condition like "if self.prev_im_pair.im_0 is None or len(self.rgbd_hist_1)<self.camera_frame_rate*self.data.time:",
            #but it guarantees regular timestamps. For example, while the aforementioned condition will guarantee that we get 90 frames in 1sec for a camera framerate of 90, we would only get
            #416 frames with the condition below. This is because 1/90=0.011111... and assuming that mujoco is running at 500 hz,  self.data.time is incremented by 1/500=0.002 everytime. The first
            #timestep N when the condition delta_time>=0.01111... becomes true is when N*0.002>=0.01111... , which implies N=6 and z=N*0.002=0.012. Since we now consider the image timestampe to be this 
            #z, and start counting from there, the condition delta_time>0.01111... well again become true when N=12, and so on. So, in 1 second, we will have 1/0.012=83.3333... frames, which is lower 
            #than our desired frame rate. Anyway, I think it's better to have regularly spaced timestamps rather than a forced number of frames with irregualr timestamps
            if self.prev_im_pair.im_0 is None or delta_time>=1.0/self.camera_frame_rate:
                rgbd_0=self.rgbd_inputs(self.data,"cam_0").astype(_default_dtype)
                rgbd_1=self.rgbd_inputs(self.data,"cam_1").astype(_default_dtype)
                if self.log_options["cams"]:
                    self.rgbd_hist_0.append(rgbd_0)
                    self.rgbd_hist_1.append(rgbd_1)

                self.prev_im_pair.im_0=rgbd_0.copy()
                self.prev_im_pair.im_1=rgbd_1.copy()
                self.prev_im_pair.ts=self.data.time
            else:
                rgbd_0=self.prev_im_pair.im_0.copy()
                rgbd_1=self.prev_im_pair.im_1.copy()

        #body states
        body_id = self.model.body("base").id  
        position = self.data.xpos[body_id].copy().astype(_default_dtype)
        orientation = quaternion.quaternion(*self.data.xquat[body_id].copy())
        rot_vec=quaternion.as_rotation_vector(orientation).astype(_default_dtype)

        #angular velocities of joints
        motor_state=np.array([self.data.qvel[self.model.joint(f"wheel_joint_{motor_idx}").id] for motor_idx in range(3)]).astype(_default_dtype)
        motor_state/=10#just to normalize
        
        motor_state=np.clip(motor_state,a_min=-2.0,a_max=2.0)
        self.prev_motor_state=motor_state.copy()

        #compute velocity
        vel=(position-self.prev_pos)/self.opt_timestep if self.prev_pos is not None else np.zeros_like(position)
        vel=np.clip(vel,a_min=-2.0,a_max=2.0)
        self.prev_pos=position.copy()
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
            angular_vel=np.clip(vee(W)/self.opt_timestep,a_min=-2.0,a_max=2.0).astype(_default_dtype)
        else:
            angular_vel=np.zeros_like(rot_vec)
        self.prev_orientation=orientation.copy()

        
        if self.disable_cameras:
            obs_cur={"orientation":rot_vec, "angular_vel": angular_vel, "vel":vel, "motor_state": motor_state, "actions": last_ctrl}
            obs=obs_cur
        else:
            obs={
                    "orientation":rot_vec,
                    "angular_vel": angular_vel,
                    "vel":vel,
                    "motor_state": motor_state,
                    "actions": last_ctrl,
                    "rgbd_0":rgbd_0.transpose(2,0,1),
                    "rgbd_1":rgbd_1.transpose(2,0,1),
                    "relative_image_timestamp": np.array([self.data.time-self.prev_im_pair.ts]).astype(_default_dtype),
                    }

        return obs
    
    def _get_info(self):
        
        position = self.data.xpos[self.model.body("base").id].copy().astype(_default_dtype)
        info={"success":False,"failure":False,"step_counter":self.step_counter, "pos2d":position[:-1]}

        return info
   
    def step(self, omniwheel_commands):
        """
        omniwheel_commands  np.tensor of shape(3,)
        """

        if self.data.time==0.0 and self.step_counter>0.0:#reset done in GUI
            print("RESET DUE TO RESET FROM GUI")
            self.reset()
        
        ctrl=omniwheel_commands*10
        ctrl=np.clip(ctrl,a_min=-10,a_max=10)#in case of pid issues

        self.data.ctrl[:] = - ctrl
        mujoco.mj_step(self.model, self.data)
        
        self.data.xfrc_applied[self.model.body("base").id, :3] = np.zeros(3)#this is to reset the initial force that is applied. From the documentation,
                                                                            #the force will be applied unless it is reset

        obs=self._get_obs(omniwheel_commands.astype(_default_dtype))
        info=self._get_info()
        terminated=False
        truncated=False

        reward=self.reward_obj(obs)/100#regularization and survival are added later

        self.reward_term_1_hist.append(reward)
        
        action_regularization=-0.0001*(np.linalg.norm(omniwheel_commands)**2)
        self.reward_term_2_hist.append(action_regularization)
        reward+=action_regularization
     
        if self.passive_viewer:

             
            with self.passive_viewer.lock():

                if 0:
                    self.passive_viewer.user_scn.ngeom=1#yeah, =1, not +=1
                    factor=20#just for display
                    hh=0.5
                    mujoco.mjv_initGeom(
                            self.passive_viewer.user_scn.geoms[self.passive_viewer.user_scn.ngeom-1],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.1]*3,
                            pos=[self.goal_2d[0]*factor,self.goal_2d[1]*factor,hh],
                            mat=np.eye(3).flatten(),
                            rgba=[1, 0, 1, 1])

                    mujoco.mjv_connector(
                            self.passive_viewer.user_scn.geoms[self.passive_viewer.user_scn.ngeom-1],
                            type=mujoco.mjtGeom.mjGEOM_LINE,
                            width=200,
                            from_=[0,0,hh],
                            to=[self.goal_2d[0]*factor,self.goal_2d[1]*factor,hh])
 
            self.passive_viewer.sync()

        self.prev_data_time=self.data.time#to detect resets that happen in GUI
        self.step_counter+=1

        if self.step_counter>=self.max_ep_steps:
            if self.verbose:
                print(f"terminated. Cause: {self.step_counter}>self.max_ep_steps")
            terminated=True

        #compute angle with upright vector
        gravity=np.array([0,0,-1.0]).astype(_default_dtype).reshape(3,1)
        gravity_local=(quaternion.as_rotation_matrix(quaternion.from_rotation_vector(obs["orientation"][-3:])).T @ (gravity)).reshape(3)

        up_axis_local=np.array([0,0,1]).astype(_default_dtype)
        angle_in_degrees=np.arccos(up_axis_local.dot(-gravity_local)).item()*180/np.pi
       
        max_allowed_tilt=20
        if angle_in_degrees>max_allowed_tilt:

            if self.verbose:
                print(f"failure after {self.step_counter}. Reason: tilt_angle > {max_allowed_tilt} ")

            info["success"]=False
            info["failure"]=True
            terminated=True
        else:
            reward+=0.02

        gamma=1.0#not used algorithmically here anyway
        self.G_tau+=(gamma**self.step_counter)*reward
        if terminated:
            if self.verbose:
                print(colored(f"G_tau=={self.G_tau}, num_steps=={self.step_counter}","magenta",attrs=["bold"]))
            self._save_logs()

        return obs, reward, terminated, truncated, info


    def close(self):

        if not self.disable_cameras:
            self.rgbd_inputs.close()

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
            max_ep_steps=args.max_steps)

    obs, _=sim.reset()
    for step_i in range(sim.max_ep_steps):

        obs, reward, terminated, _, info=sim.step(sim.action_space.sample())

