import numpy as np
from gym.spaces.box import Box
from gym.core import Wrapper, ObservationWrapper, RewardWrapper


class RewardTimeIncentiver(RewardWrapper):
    """
    Gym wrapper that applies a constant discount on every step in order to
    incentive the agent to finish as soon as possible its tasks
    __init__
        :param env: gym environment (gym.env)
        :param eps: epsilon reward substracted on each step (float)
    """
    def __init__(self, env, eps):
        super(RewardTimeIncentiver, self).__init__(env)
        self.eps = eps

    def reward(self, rew, eps=0.01):
        return rew - self.eps


class BreakOutPreprocessor(ObservationWrapper):
    """
    Gym wrapper that preprocess the observations of the Atari Breakout environment
    by performing two operations:
    - Color scaling to -1 and 1
    - Image size scaling and cropping
    __init__
        :param env: gym environment (gym.env)
    """
    def __init__(self, env):
        super(BreakOutPreprocessor, self).__init__(env)

    def observation(self, obs):
        obs = (obs / 200 - 0.5) * 2
        obs = obs[20:-14:2, ::2]
        return obs


class FrameBuffer(Wrapper):
    """
    Gym wrapper that accumulates sequential observations of an environment in the
    channels dimension, in order to make an environment that needs it meet the
    Markov property
    __init__
        :param env: gym environment (gym.env)
        :param n_frames: number of frames to accumulate in the buffer (int)
        :param dim_order: "tensorflow" or "pytorch". The former leaves the channels
        in the last dimension, the last one leaves them in the first dimension.
    """
    def __init__(self, env, n_frames=4, dim_order='tensorflow'):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)
        self.dim_order = dim_order
        if dim_order == 'tensorflow':
            height, width, n_channels = env.observation_space.shape
            obs_shape = [height, width, n_channels * n_frames]
        elif dim_order == 'pytorch':
            n_channels, height, width = env.observation_space.shape
            obs_shape = [n_channels * n_frames, height, width]
        else:
            raise ValueError('dim_order should be "tensorflow" or "pytorch", got {}'.format(dim_order))
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info

    def update_buffer(self, img):
        if self.dim_order == 'tensorflow':
            offset = self.env.observation_space.shape[-1]
            axis = -1
            cropped_framebuffer = self.framebuffer[:, :, :-offset]
        elif self.dim_order == 'pytorch':
            offset = self.env.observation_space.shape[0]
            axis = 0
            cropped_framebuffer = self.framebuffer[:-offset]
        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis=axis)


class LimitLength(Wrapper):
    """
    Gym wrapper that limits the length of an episode
    __init__
        :param env: gym environment (gym.env)
        :param k: number of maximum steps per episode (k)
    """
    def __init__(self, env, k):
        Wrapper.__init__(self, env)
        self.k = k
        self.cnt = 0

    def reset(self):
        # This assumes that reset() will really reset the env.
        # If the underlying env tries to be smart about reset
        # (e.g. end-of-life), the assumption doesn't hold.
        ob = self.env.reset()
        self.cnt = 0
        return ob

    def step(self, action):
        ob, r, done, info = self.env.step(action)
        self.cnt += 1
        if self.cnt == self.k:
            done = True
        return ob, r, done, info
