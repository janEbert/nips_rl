import numpy as np
from osim.env import ProstheticsEnv
from gym.spaces import Box, MultiBinary


class RunEnv2(ProstheticsEnv):
    def __init__(self, state_transform, visualize=False,
                 integrator_accuracy=5e-5, model='3D', prosthetic=True,
                 difficulty=0, skip_frame=5, reward_mult=10.):
        super(RunEnv2, self).__init__(visualize, integrator_accuracy)
        self.args = (model, prosthetic, difficulty)
        self.change_model(*self.args)
        self.state_transform = state_transform
        self.observation_space = Box(-1000, 1000, [state_transform.state_size])
        self.noutput = self.get_observation_space_size()
        self.action_space = MultiBinary(self.get_action_space_size())
        self.skip_frame = skip_frame
        self.reward_mult = reward_mult

    def reset(self, difficulty=2, seed=None):
        self.change_model(self.args[0], self.args[1], difficulty, seed)
        s = super(RunEnv2, self).reset()
        self.state_transform.reset()
        s, _ = self.state_transform.process(s)
        return s

    def _step(self, action):
        action = np.clip(action, 0, 1)
        info = {'original_reward':0}
        reward = 0.
        for _ in range(self.skip_frame):
            s, r, t, _ = super(RunEnv2, self).step(action)
            info['original_reward'] += r
            s = self.state_transform.process(s)
            reward += r
            if t:
                break

        return s, reward*self.reward_mult, t, info


class JumpEnv(ProstheticsEnv):
    noutput = 9
    ninput = 38

    def __init__(self, visualize=False, integrator_accuracy=5e-5):
        super(JumpEnv, self).__init__(visualize, integrator_accuracy)
        self.action_space = MultiBinary(9)

    def get_observation(self):
        observation = super(JumpEnv, self).get_observation()
        return observation[:-3]

    def _step(self, action):
        action = np.tile(action, 2)
        #action = np.repeat(action, 2)
        s, r, t, info = super(JumpEnv, self)._step(action)
        return s, 10*r, t, info
