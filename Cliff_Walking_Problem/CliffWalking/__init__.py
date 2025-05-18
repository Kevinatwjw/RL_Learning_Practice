from CliffWalking.envs.GridWorld import CliffWalkingEnv
from CliffWalking.wrappers.Hashposotion import HashPosition
from gym.envs.registration import register
register(
    id='CliffWalking/CliffWalkingEnv-v0',
    entry_point='CliffWalking.envs:CliffWalkingEnv',
    max_episode_steps=300,
)
