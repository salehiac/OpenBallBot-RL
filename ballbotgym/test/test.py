import gymnasium as gym
import ballbotgym
import sys
import pdb
import numpy as np
import string
import cv2
import os
import argparse



def main(args):

    rand_str = ''.join(np.random.permutation(list(string.ascii_letters + string.digits))[:12])
    log_dir=f"/tmp/log_{rand_str}"
    os.mkdir(log_dir)
    
    env=gym.make(
            "ballbot-v0.1",
            GUI=True,
            goal_type=args.goal_type,
            disable_cameras=True,
            test_only=False)
    
    obs, _=env.reset()
    for step_i in range(env.env.env.max_ep_steps):
        obs, reward, terminated, _, info=env.step(env.action_space.sample()*5)
    
        if terminated:
            print(f"episode ended. Success={info['success']}")
            break
    
        #pdb.set_trace()
        if not env.env.env.disable_cameras:
            cv2.imwrite(log_dir+f"/{step_i}_rgb_0.png",   255*cv2.merge(cv2.split(obs["rgbd_0"][:,:,:3])[::-1]))
            cv2.imwrite(log_dir+f"/{step_i}_depth_0.png", 255*obs["rgbd_0"][:,:,3])
            cv2.imwrite(log_dir+f"/{step_i}_rgb_1.png",   255*cv2.merge(cv2.split(obs["rgbd_1"][:,:,:3])[::-1]))
            cv2.imwrite(log_dir+f"/{step_i}_depth_1.png", 255*obs["rgbd_1"][:,:,3])
    
    env.close()
    



if __name__=="__main__":

    _parser = argparse.ArgumentParser(description="test bbot env")
    _parser.add_argument('--goal_type', type=str, help="goal_type can either be fixed_pos, fixed_dir, rand_pos, rand_dir or stop", default="rand_dir")

    _args = _parser.parse_args()

    main(_args)

