import numpy as np
from gym.wrappers import RescaleAction
import torch


def get_env(env_name, wrap_torch=True):
    from env.torch_wrapper import TorchWrapper
    from env.hopper_no_bonus import HopperNoBonusEnv
    from env.cheetah_no_flip import CheetahNoFlipEnv
    from env.ant_no_bonus import AntNoBonusEnv
    from env.navigation1 import Navigation1
    from env.humanoid_no_bonus import HumanoidNoBonusEnv
    from env.hopper_less_bonus import HopperLessBonusEnv
    from env.cheetah_less_flip import CheetahLessFlipEnv
    from env.ant_less_bonus import AntLessBonusEnv
    from env.navigation2 import Navigation2
    from env.humanoid_less_bonus import HumanoidLessBonusEnv
    envs = {
        'hopper': HopperLessBonusEnv,
        'cheetah-no-flip': CheetahLessFlipEnv,
        'ant': AntLessBonusEnv,
        'humanoid': HumanoidLessBonusEnv,
        'navigation1':Navigation2
    }
    envs1 = {
        'hopper': HopperNoBonusEnv,
        'cheetah-no-flip': CheetahNoFlipEnv,
        'ant': AntNoBonusEnv,
        'humanoid': HumanoidNoBonusEnv,
        'navigation1':Navigation1
    }
    
    env = envs[env_name]()
    env1 = envs1[env_name]()

    if not (np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0)):
        env = RescaleAction(env, -1.0, 1.0)
    if wrap_torch:
        env = TorchWrapper(env)

    if not (np.all(env1.action_space.low == -1.0) and np.all(env1.action_space.high == 1.0)):
        env1 = RescaleAction(env1, -1.0, 1.0)
    if wrap_torch:
        env1 = TorchWrapper(env1)

    return env,env1