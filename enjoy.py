import argparse
import os
import types

import numpy as np
import torch

from vec_env.dummy_vec_env import DummyVecEnv
#from vec_env.vec_normalize import VecNormalize
from envs import VecPyTorch, make_vec_envs
os.sys.path.insert(0, os.path.abspath('settings_folder'))
from storage import RolloutStorage

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/ppo',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--text-policy', action='store_true', default=True,
                    help='use a text policy')
parser.add_argument('--num-processes', type=int, default=16,
                    help='how many training CPU processes to use')
parser.add_argument('--num-steps', type=int, default=5,
                    help='number of forward steps in A2C (default: 5)')
parser.add_argument('--recurrent-policy', action='store_true', default=False,
                    help='use a recurrent policy')

args = parser.parse_args()

env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                            None, None, args.add_timestep, device='cpu', allow_early_resets=False)

# Get a render function
f = open("cFile.txt", "w")

render_func = None
tmp_env = env
while True:
    if hasattr(tmp_env, 'envs'):
        render_func = tmp_env.envs[0].render
        break
    elif hasattr(tmp_env, 'venv'):
        tmp_env = tmp_env.venv
    elif hasattr(tmp_env, 'env'):
        tmp_env = tmp_env.env
    else:
        break

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

actor_critic.eval()

"""
if isinstance(env.venv, VecNormalize):
    env.venv.ob_rms = ob_rms

    # An ugly hack to remove updates
    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    env.venv._obfilt = types.MethodType(_obfilt, env.venv)
"""

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

rollouts = RolloutStorage(args.num_steps, args.num_processes,
                    env.observation_space, env.action_space,
                    actor_critic.recurrent_hidden_state_size)

# if render_func is not None:
#     render_func('human')

obs = env.reset()

rollouts.obs["image"][0].copy_(obs.image)
rollouts.obs["text"][0].copy_(obs.text)
#print("after rollout reset", obs)
#rollouts.to(device)
if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

while True:
    for step in range(args.num_steps):
        with torch.no_grad():
            obs_dict = {"image": rollouts.obs["image"][step], "text": rollouts.obs["text"][step]}
            #print("obs_dict",obs_dict["text"])
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs_dict, recurrent_hidden_states, masks, deterministic=False,)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    # if render_func is not None:
    #     render_func('human')
