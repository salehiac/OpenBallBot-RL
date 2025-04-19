import gymnasium as gym
import ballbotgym
import sys
import pdb


env=gym.make(
        "ballbot-v0.1",
        GUI=True,
        max_ep_steps=10000,
        apply_random_force_at_init=True)

obs, _=env.reset()
for step_i in range(env.env.env.max_ep_steps):
    obs, reward, terminated, _, info=env.step(env.action_space.sample()*5)

    if terminated:
        print(f"episode ended. Success={info['success']}")
        break

env.close()





