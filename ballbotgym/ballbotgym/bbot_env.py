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

from . import Rewards, terrain


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
        self.max_ep_steps=4000

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

        elif goal_type=="fixed_dir":#no position, and no need for goal conditioning

            self.window=3
            self.observation_space=gym.spaces.Dict({
                "orientation": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float64),
                "angular_vel": gym.spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float64),
                "vel": gym.spaces.Box(low=-2,high=2, shape=(3,), dtype=np.float64),
                "rgbd_0": gym.spaces.Box(low=0.0, high=1.0, shape=(im_shape["h"],im_shape["w"], 4), dtype=np.float64),
                "rgbd_1": gym.spaces.Box(low=0.0, high=1.0, shape=(im_shape["h"],im_shape["w"], 4), dtype=np.float64),
                }) if not disable_cameras else gym.spaces.Dict({
                    "orientation": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(3*self.window,), dtype=np.float64),
                    "angular_vel": gym.spaces.Box(low=-2, high=2, shape=(3*self.window,), dtype=np.float64),
                    "vel": gym.spaces.Box(low=-2,high=2, shape=(3*self.window,), dtype=np.float64),
                    "motor_state": gym.spaces.Box(-2.0,2.0,shape=(3*self.window,),dtype=np.float64),
                    "actions": gym.spaces.Box(-1.0,1.0,shape=(3*self.window,),dtype=np.float64),
                    })

        elif goal_type=="stop" or goal_type=="rand_pos": 
            
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


        self.reward_term_1_hist=[]#
        self.reward_term_2_hist=[]
        self.reward_term_3_hist=[]#constant for now

    @property
    def opt_timestep(self):
        return self.model.opt.timestep

    def _reset_goal_and_reward_objs(self):
        
        if self.goal_type=="rand_dir":
                
            self.goal_2d=sample_direction_uniform(num=1).reshape(2)
            self.reward_obj=Rewards.DirectionalReward(target_direction=self.goal_2d)

        elif self.goal_type=="fixed_dir":

            self.goal_2d=[0.0, 1.0]
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
            self.action_hist=[]

        print("goal_2d==",self.goal_2d)

    def _reset_terrain(self):

        nrows=self.model.hfield_nrow.item()
        ncols=self.model.hfield_ncol.item() 

        assert nrows==ncols, f"terrain is expected to have an equal number of rows an cols (got {nrows}, {ncols} in xml file)"
        assert self.model.hfield_size[0,0]==self.model.hfield_size[0,1], f"terrain should have equal length and width (got {self.model.hfield_size[0,:2]} in xml file)"

        sz=self.model.hfield_size[0,0]
        hfield_height_coef=self.model.hfield_size[0,2]

        self.model.hfield_data=terrain.generate_perlin_terrain(nrows,seed=np.random.randint(10000))
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

        print("resetting_env...")

        super().reset(seed=seed)

        self._reset_goal_and_reward_objs()

        self.step_counter=0
        self.prev_data_time=0

        init_robot_height_offset=self._reset_terrain()
      
        mujoco.mj_resetData(self.model, self.data)
        self.data.joint("base_free_joint").qpos[2]+=init_robot_height_offset
        self.data.joint("ball_free_joint").qpos[2]+=init_robot_height_offset
        mujoco.mj_forward(self.model, self.data) #recompute derivatives etc

        #pdb.set_trace()

        self.prev_pos=None
        self.prev_orientation=None
        self.prev_motor_state=None


        #each of those will be of length (self.window-1)*relevant_dim, since the latest obs (of size relevant_dim) will be concatenated to it by _get_obs
        self.obs_window={
                "orientation": np.zeros((self.window-1)*3),
                "angular_vel": np.zeros((self.window-1)*3),
                "vel": np.zeros((self.window-1)*3),
                "motor_state": np.zeros((self.window-1)*3),
                "actions": np.zeros((self.window-1)*3),
                }
       
        obs=self._get_obs(np.zeros(3))
        info=self._get_info()

        if not self.disable_cameras:
            for rgbd_input in self.rgbd_inputs:
                rgbd_input.reset(self.model)
        if self.scene_renderer is not None:
            self.scene_renderer.reset(self.model,self.num_resets)

        self.G_tau=0.0
       
        self.num_resets+=1

        if len(self.reward_term_1_hist):
            np.save(self.log_dir+"/term_1",np.array(self.reward_term_1_hist))
            np.save(self.log_dir+"/term_2",np.array(self.reward_term_2_hist))

        return obs, info

    def _get_obs(self,last_ctrl):

        if not self.disable_cameras:
            rgbd_0=self.rgbd_inputs[0](self.data).astype("float64")
            rgbd_1=self.rgbd_inputs[1](self.data).astype("float64")

        #body states
        body_id = self.model.body("base").id  
        position = self.data.xpos[body_id].copy().astype("float64")
        orientation = quaternion.quaternion(*self.data.xquat[body_id].copy())
        rot_vec=quaternion.as_rotation_vector(orientation).astype("float64")

        #angular velocities of joints
        motor_state=np.array([self.data.qvel[self.model.joint(f"wheel_joint_{motor_idx}").id] for motor_idx in range(3)])
        motor_state/=10#just to normalize
        if any(np.abs(motor_state)>2):
            print("WARNING!!!!!!!! =========================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(motor_state)
        motor_state=np.clip(motor_state,a_min=-2.0,a_max=2.0)
        #motor_acc=(motor_state-self.prev_motor_state)/self.opt_timestep if self.prev_motor_state is not None else np.zeros_like(motor_state)
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
            angular_vel=np.clip(vee(W)/self.opt_timestep,a_min=-2.0,a_max=2.0)
        else:
            angular_vel=np.zeros_like(rot_vec)
        self.prev_orientation=orientation.copy()

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

        elif self.goal_type=="fixed_dir":
            
            if self.disable_cameras:
                obs_cur={"orientation":rot_vec, "angular_vel": angular_vel, "vel":vel, "motor_state": motor_state, "actions": last_ctrl}
                assert self.window==3, "hardcoded for a size 3 sliding window"
                for key in obs_cur.keys():
                    dim=obs_cur[key].shape[0]
                    obs_cur[key]=np.hstack([self.obs_window[key], obs_cur[key]])

                    self.obs_window[key][:dim]=self.obs_window[key][dim:2*dim]
                    self.obs_window[key][dim:2*dim]=obs_cur[key][2*dim:]

                obs=obs_cur
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

        if self.test_only:
            self.action_hist.append(ctrl.reshape(1,3))
       
        self.data.ctrl[:] = - ctrl
        mujoco.mj_step(self.model, self.data)
        
        self.data.xfrc_applied[self.model.body("base").id, :3] = np.zeros(3)#this is to reset the initial force that is applied. From the documentation,
                                                                            #the force will be applied unless it is reset

        if self.scene_renderer is not None:
            self.scene_renderer(self.data)

        obs=self._get_obs(omniwheel_commands)
        info=self._get_info()
        terminated=False
        truncated=False

        reward=self.reward_obj(obs)#note that failure penalties are added later
        reward= reward/1000 if self.goal_type=="rand_dir" else reward/100 if self.goal_type=="fixed_dir" else reward/100 #normalization to get better gradients

        self.reward_term_1_hist.append(reward)
        
        #pdb.set_trace() 
        action_regularization=-0.001*(np.linalg.norm(omniwheel_commands)**2)
        #print(action_regularization)
        self.reward_term_2_hist.append(action_regularization)
        reward+=action_regularization
     
        if self.passive_viewer:

             
            with self.passive_viewer.lock():

                if self.goal_type in ["rand_dir", "fixed_dir"]:
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
        gravity_local=(quaternion.as_rotation_matrix(quaternion.from_rotation_vector(obs["orientation"][-3:])).T @ (gravity)).reshape(3)

        up_axis_local=np.array([0,0,1]).astype("float")
        angle_in_degrees=np.arccos(up_axis_local.dot(-gravity_local)).item()*180/np.pi
       
        max_allowed_tilt=20
        early_fail_penalty=0.0
        if angle_in_degrees>max_allowed_tilt:
            print(f"failure after {self.step_counter}. Reason: tilte_angle > {max_allowed_tilt} ")

            info["success"]=False
            info["failure"]=True
            terminated=True
            #early_fail_penalty=-1.0
            #reward+=early_fail_penalty
        else:
            reward+=0.01

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
                if 0:
                    import matplotlib.pyplot as plt
                    aa=np.concatenate(self.action_hist,0)
                    plt.plot(aa[:,0],"r")
                    plt.plot(aa[:,1],"g")
                    plt.plot(aa[:,2],"b")
                    #plt.plot(self.reward_hist,"b")
                    plt.show()

                    plt.plot(self.reward_term_1_hist,"r")
                    plt.plot(self.reward_term_2_hist,"g")
                    plt.show()


        if self.test_only:
            self.reward_hist.append(reward)


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
            max_ep_steps=args.max_steps)

    obs, _=sim.reset()
    for step_i in range(sim.max_ep_steps):

        obs, reward, terminated, _, info=sim.step(sim.action_space.sample())





if __name__=="__main__":

    _parser = argparse.ArgumentParser(description="bbotgym test")
    _parser.add_argument("--xml_path", type=str)
    _parser.add_argument("--gui", action="store_true")

    _args = _parser.parse_args()
    main(_args)
