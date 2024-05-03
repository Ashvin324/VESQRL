from .hopper import HopperEnv
import numpy as np

class HopperNoBonusEnv(HopperEnv):
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        info['violation']=done
        reward -= 1     # subtract out alive bonus
        return next_state, reward, done, info

    def check_violation(self,states):
        heights,angs=states[:,0],states[:,1]
        return ~(np.isfinite(states).all(axis=1) & (np.abs(states[:,1:]) < 100).all(axis=1) & (heights > .7) & (np.abs(angs) < .2))
    def check_done(self,states):
        return self.check_violation(states)
