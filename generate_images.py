import numpy as np
import env
import gym
import time
import cv2
import pickle as pkl

num_samples = 1000
num_objects = 3
goal_dim = num_objects * (num_objects - 1) * 3 // 2
env_name = 'FetchManipulate{}Objects-v0'.format(num_objects)

environment = gym.make(env_name)
obs = environment.reset()
configs = []
for i in range(num_samples):
    if i % 100 == 0:
        print(i)
    environment.unwrapped.reset_goal(goal=np.zeros([goal_dim]), biased_init=True)
    for u in range(5):
        obs, _, _, _ = environment.step([0, 0, 0, 0])
    img = environment.sim.render(camera_name='main1', width=128, height=128, depth=False)
    img = np.flip(img)
    cv2.imwrite("images/{0}.png".format(i+1), img)
    configs.append(obs['achieved_goal'])
environment.close()

with open('configurations.pkl', 'wb') as f:
    pkl.dump(np.array(configs), f)