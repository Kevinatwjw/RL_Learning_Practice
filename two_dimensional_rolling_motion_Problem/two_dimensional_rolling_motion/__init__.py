from two_dimensional_rolling_motion.wrappers.Hashposotion import DiscreteActionWrapper
from two_dimensional_rolling_motion.wrappers.Hashposotion import FlattenActionSpaceWrapper
from two_dimensional_rolling_motion.envs.GridWorld import RollingBall
from gym.envs.registration import register
register(
    id='CliffWalking/CliffWalkingEnv-v0',
    entry_point='CliffWalking.envs:CliffWalkingEnv',
    max_episode_steps=300,
)
