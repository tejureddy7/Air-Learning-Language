import os

import gym
import numpy as np
import torch
import torch_ac
import re
from gym.spaces.box import Box

from gym import spaces

from vec_env import VecEnvWrapper
from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subproc_vec_env import SubprocVecEnv

try:
    import gym_airsim
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, add_timestep, allow_early_resets):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)

        obs_shape = env.observation_space.spaces["image"].shape

        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        #if log_dir is not None:
        #    env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
        #                        allow_early_resets=allow_early_resets)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.spaces["image"].shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env

    return _thunk

def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep, device, allow_early_resets):
    envs = [make_env(env_name, seed, i, log_dir, add_timestep, allow_early_resets) for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    if len(envs.observation_space.spaces["image"].shape) == 3:
        print('Creating frame stacking wrapper')
        envs = VecPyTorchFrameStack(envs, 4, device)


    return envs


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.spaces["image"].low[0],
            self.observation_space.spaces["image"].high[0],
            [self.observation_space.spaces["image"].shape[0] + 1],
            dtype=self.observation_space.spaces["image"].dtype)

        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.spaces["image"].shape
        self.observation_space = Box(
            self.observation_space.spaces["image"].low[0, 0, 0],
            self.observation_space.spaces["image"].high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.spaces["image"].dtype)

        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

    def observation(self, observation):

        observation["image"] = observation["image"].transpose(2, 1, 0)

        return observation


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):



        obss = self.venv.reset()


        obs_space = {"image": obss[0]["image"].shape, "text": 100}
        vocab = Vocabulary(obs_space["text"])

        ''' a list of dictionaries
        imgs = self.preprocess_images([obs["image"] for obs in obss])
        txts = self.preprocess_texts([obs["mission"] for obs in obss], vocab)
        obs = [ {"image": imgs[i], "text": txts[i]} for i in range(len(obss))]
        '''

        obs = torch_ac.DictList({
            "image": self.preprocess_images([obs["image"] for obs in obss]),
            "text": self.preprocess_texts([obs["mission"] for obs in obss], vocab)
        })


        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):

        obss, reward, done, info = self.venv.step_wait()
        obs_space = {"image": obss[0]["image"].shape, "text": 100}
        vocab = Vocabulary(obs_space["text"])
        obs = torch_ac.DictList({
            "image": self.preprocess_images([obs["image"] for obs in obss]),
            "text": self.preprocess_texts([obs["mission"] for obs in obss], vocab)
        })

        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        return obs, reward, done, info

    def preprocess_images(self, images, device=None):
        # Bug of Pytorch: very slow if not first converted to numpy array
        images = np.array(images)
        return torch.tensor(images, device=device, dtype=torch.float)

    def preprocess_texts(self, texts, vocab, device=None):

        var_indexed_texts = []
        max_text_len = 0

        for text in texts:
            tokens = re.findall("([a-z]+)", text.lower())
            var_indexed_text = np.array([vocab[token] for token in tokens])
            var_indexed_texts.append(var_indexed_text)
            max_text_len = max(len(var_indexed_text), max_text_len)

        indexed_texts = np.zeros((len(texts), max_text_len))

        for i, indexed_text in enumerate(var_indexed_texts):
            indexed_texts[i, :len(indexed_text)] = indexed_text

        return torch.tensor(indexed_texts, device=device, dtype=torch.long)



class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space.spaces["image"]  # wrapped ob space
        self.shape_dim0 = wos.low.shape[0]
        self.shape_dim1 = 5
        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)
        self.stackedobs_img = np.zeros((venv.num_envs,) + low.shape)

        self.stackedobs_txt = np.zeros((venv.num_envs,) + (5,))
        self.stackedobs = {"image": torch.from_numpy(self.stackedobs_img).float(), "text": torch.from_numpy(self.stackedobs_txt).long()}
        self.stackedobs = {"image": self.stackedobs["image"].to(device), "text": self.stackedobs["text"].to(device)}
        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.spaces["image"].dtype)
        observation_space = spaces.Dict({
            'image': observation_space
        })
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs["image"][:, :-self.shape_dim0] = self.stackedobs["image"][:, self.shape_dim0:].clone()
        self.stackedobs["text"][:, :-self.shape_dim0] = self.stackedobs["text"][:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs["image"][i] = 0
                self.stackedobs["text"][i] = 0

        self.stackedobs["image"][:, -self.shape_dim0:] = obs.image
        self.stackedobs["text"][:, -self.shape_dim1:] = obs.text
        return self.stackedobs, rews, news, infos

    def reset(self):

        obs = self.venv.reset()
        #print("hhhhhh",self.stackedobs["image"])

        self.stackedobs["image"].fill_(0)
        self.stackedobs["text"].fill_(0)


        self.stackedobs["image"][:, -self.shape_dim0:] = obs.image
        self.stackedobs["text"][:, -self.shape_dim1:] = obs.text

        return torch_ac.DictList(self.stackedobs)

    def close(self):
        self.venv.close()
