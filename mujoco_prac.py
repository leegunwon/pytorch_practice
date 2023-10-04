import random
from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv

env = HalfCheetahEnv()

for _ in range(1000):
	env.render()
	action = [random.randint(0,1) for _ in range(env.action_space.shape[0])]
	_, _, _, _ = env.step(action)

env.close()