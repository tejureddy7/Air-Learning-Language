import sys
import gym

import os
import tensorflow as tf
os.sys.path.insert(0, os.path.abspath('settings_folder'))
import settings
import msgs
from gym_airsim.envs.airlearningclient import *
import callbacks
# from multi_modal_policy import MultiInputPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines import DQN
from stable_baselines.deepq import DQN, MlpPolicy, CnnPolicy
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import EvalCallback

from keras.backend.tensorflow_backend import set_session
from tensorboard import summary

from keras.callbacks import TensorBoard
#from stable_baselines.common.callbacks import CheckpointCallback
# from datetime import datetime
# #from packaging import version
#
# import tensorflow as tf
# from tensorflow import keras
#
# import numpy as np




def setup(difficulty_level='default', env_name = "AirSimEnv-v42"): #add weights for further training
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    sess = set_session(tf.Session(config=config))
    msgs.algo = "DQN-B"

    env = gym.make(env_name)
    #env.init_again(eval("settings."+difficulty_level+"_range_dic"))

    #agent = DQN.load(weights, tensorboard_log="./ppo2_circledetect_new/" ) #load moddel for further training

    # Vectorized environments allow to easily multiprocess training
    # we demonstrate its usefulness in the next examples

    vec_env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    agent = DQN(CnnPolicy, vec_env, verbose=1, tensorboard_log="./circledetect_dqn/") #tej add tensorflow log #image_detect has the dqn log


    print(agent.policy)

    #agent.set_env(vec_env) #for further training

    env.set_model(agent)


    return env, agent

def train(env, agent):

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./dqnnewlogs10/',
                                                 name_prefix='rl_model')

    # Train the agent
    agent.learn(total_timesteps=100000 , tb_log_name="dqn1sphere", reset_num_timesteps=False, callback = checkpoint_callback ) #tej subfolder


    agent.save("dqn1spheregray",cloudpickle=True) #dqnnew 90 #dqnnew1 fail #ddqnnew2 continue on collision # dqnsimmtorealtest7 324,244 img size

def test(env, agent, filepath):
    model = DQN.load(filepath)
    obs = env.reset()
    episode_count = 0
    while (True):
        if(episode_count == settings.testing_nb_episodes_per_model):
            exit(0)
        else:
            action, _states = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            if(dones == True):
                env.reset()
                episode_count += 1

if __name__ == "__main__":
    env, agent = setup()
    train(env,agent)

    # model_weights_list_to_test = ["C:\\Users\\EEHPC\\Airlearning_project2\\airlearning-rl2\\dqn2sphere1.pkl"]
    # #
    # #
    # #
    # #
    # task = {"weights": model_weights_list_to_test}
    # #print(len(task["weights"]))
    # for weights in task["weights"]:
    #     #print("weights",weights)
    #     #utils.reset_msg_logs()
    #     env, agent = setup(weights)
    #     #msgs.mode = 'test'
    #     #test(env, agent, weights)
    #     train(env,agent)
