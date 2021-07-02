import numpy as np
import env
import gym
import os
import cv2
import pickle as pkl

num_samples = 38000
num_objects = 3
goal_dim = num_objects * (num_objects - 1) * 3 // 2
env_name = 'FetchManipulate{}Objects-v0'.format(num_objects)

environment = gym.make(env_name)
obs = environment.reset()
n_configs = 0
configs_to_id = {}
for i in range(num_samples):
    # Create logs and models folder if they don't exist
    if not os.path.exists('./images/train'):
        os.makedirs('./images/train')
    if not os.path.exists('./images/test'):
        os.makedirs('./images/test')
    if i < num_samples * 3 // 4:
        split = 'train'
    else:
        split = 'test'
    if i % 1000 == 0:
        print(i)
    environment.unwrapped.reset_goal(goal=np.zeros([goal_dim]), biased_init=True)
    for u in range(5):
        obs, _, _, _ = environment.step([0, 0, 0, 0])
    img = environment.sim.render(camera_name='main1', width=128, height=128, depth=False)
    img = np.flip(img)
    ag = obs['achieved_goal']
    if str(ag) not in configs_to_id.keys():
        n_configs += 1
        configs_to_id[str(ag)] = n_configs
    if not os.path.exists('./images/{}/{}'.format(split, configs_to_id[str(ag)])):
        os.makedirs('./images/{}/{}'.format(split, configs_to_id[str(ag)]))
    cv2.imwrite("images/{0}/{1}/{2}.jpg".format(split, configs_to_id[str(ag)], i+1), img)
environment.close()

# with open('configurations.pkl', 'wb') as f:
#     pkl.dump(np.array(configs), f)