from gym.envs.registration import register

import sys
from functools import reduce


def str_to_class(str):
    return reduce(getattr, str.split("."), sys.modules[__name__])


# Robotics
# ----------------------------------------
num_blocks = 3
kwargs = {'reward_type': 'sparse'}

register(id='FetchManipulate3Objects-v0',
         entry_point='env.envs:FetchManipulateEnv',
         kwargs=kwargs,
         max_episode_steps=100,)

register(id='FetchManipulate3ObjectsAtomic-v0',
         entry_point='env.envs:FetchManipulateEnvAtomic',
         kwargs=kwargs,
         max_episode_steps=100,)

register(id='FetchManipulate4ObjectsAtomic-v0',
         entry_point='env.envs:FetchManipulateEnvAtomic',
         kwargs={'reward_type': 'sparse',
                 'num_blocks': 4,
                 'model_path': 'fetch/stack4.xml'
                 },
         max_episode_steps=100,)

register(id='FetchManipulate5ObjectsAtomic-v0',
         entry_point='env.envs:FetchManipulateEnvAtomic',
         kwargs={'reward_type': 'sparse',
                 'num_blocks': 5,
                 'model_path': 'fetch/stack5.xml'
                 },
         max_episode_steps=100,)
