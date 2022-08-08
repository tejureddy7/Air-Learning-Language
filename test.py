import sys
import gym
import os
import numpy as np
import time
import random
os.sys.path.insert(0, os.path.abspath('settings_folder'))
import settings
import msgs
from gym_airsim.envs.airlearningclient import *
from itertools import islice



#msgs.algo = "PPO"
msgs.algo = "DQN-B"

env_name = "AirSimEnv-v42"
difficulty_level='default'



env = gym.make(env_name)

#for i in range(5):
    #env.init_again(eval("settings."+difficulty_level+"_range_dic"))

# print("action space:",env.action_space)
# print("action space shape: ",env.action_space.shape)
#
# print("observation_space:",env.observation_space)
# print("observation_space shape:",env.observation_space.shape)
#
# obs = env.reset()
#
for _ in range(15):
    #obs = env.reset()

    action1 = 0 #env.action_space.sample()
    action2 = 2


    #env.step(action1)
    for _ in range(20):
        #env.step(action2)
        env.step(action1)
        env.step(action2)



#env.close()
